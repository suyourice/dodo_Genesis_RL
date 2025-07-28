# Python- und Genesis-Importe
import torch
import os
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# ---------------------------------------------------
# Reward Registry
# ---------------------------------------------------
REWARD_REGISTRY = {}

def register_reward():
    """Dekorator für Reward-Methoden; der Key wird automatisch aus dem Methodennamen abgeleitet."""
    def wrap(fn):
        key = fn.__name__.removeprefix("_reward_")
        REWARD_REGISTRY[key] = fn
        return fn
    return wrap

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class DodoEnv:
    CONTACT_HEIGHT = 0.05
    SWING_HEIGHT_THRESHOLD = 0.10

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.device = gs.device
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.command_cfg = command_cfg
        self.reward_cfg = reward_cfg
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_commands = command_cfg["num_commands"]
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.obs_scales = obs_cfg.get("obs_scales", {})
        self.reward_scales = reward_cfg.get("reward_scales", {})

        # === Rewards vorbereiten ===
        self.reward_functions = {}
        for name, scale in self.reward_scales.items():
            if name not in REWARD_REGISTRY:
                raise KeyError(f"Reward '{name}' nicht implementiert.")
            fn = REWARD_REGISTRY[name].__get__(self, type(self))
            self.reward_functions[name] = fn
            self.reward_scales[name] = scale
        self.episode_sums = {name: torch.zeros((self.num_envs,), device=self.device) for name in self.reward_scales}

        # === Szene & Roboter ===
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane(fixed=True))

        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=env_cfg.get("robot_mjcf", "dodo.xml"),
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        self.scene.build(n_envs=num_envs)

        # === Nach scene.build(): Gelenke und Kräfte setzen ===
        self.motors_dof_idx = [self.robot.get_joint(n).dof_start for n in env_cfg["joint_names"]]

        kp = [env_cfg["kp"]] * self.num_actions
        kd = [env_cfg["kd"]] * self.num_actions
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx)

        self.robot.set_dofs_force_range(
            lower=-env_cfg.get("clip_actions", 100.0) * torch.ones(self.num_actions, dtype=torch.float32),
            upper= env_cfg.get("clip_actions", 100.0) * torch.ones(self.num_actions, dtype=torch.float32),
            dofs_idx_local=self.motors_dof_idx,
        )

        self.ankle_links = [self.robot.get_link(n) for n in env_cfg.get("foot_link_names", [])]
        self.hip_aa_indices = [env_cfg["joint_names"].index("Left_HIP_AA"), env_cfg["joint_names"].index("Right_HIP_AA")]
        self.hip_fe_indices = [env_cfg["joint_names"].index("Left_THIGH_FE"), env_cfg["joint_names"].index("Right_THIGH_FE")]
        self.knee_fe_indices = [env_cfg["joint_names"].index("Left_KNEE_FE"), env_cfg["joint_names"].index("Right_SHIN_FE")]

        # === Initialisiere Beobachtungs- und Aktionsspeicher ===
        self._init_buffers()

        self.commands[:] = gs_rand_float(
            self.command_cfg["command_ranges"]["lin_vel_x"][0],
            self.command_cfg["command_ranges"]["lin_vel_x"][1],
            (self.num_envs, self.num_commands),
            self.device,
        )


    def _init_buffers(self):
        N, A, C = self.num_envs, self.num_actions, self.num_commands
        self.base_lin_vel = torch.zeros((N, 3), device=self.device)
        self.base_ang_vel = torch.zeros((N, 3), device=self.device)
        self.projected_gravity = torch.zeros((N, 3), device=self.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N,1)
        self.obs_buf = torch.zeros((N, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((N,), device=self.device)
        self.reset_buf = torch.ones((N,), dtype=torch.int32, device=self.device)
        self.episode_length_buf = torch.zeros((N,), dtype=torch.int32, device=self.device)
        self.commands = torch.zeros((N, C), device=self.device)
        self.commands_scale = torch.tensor([
            self.obs_scales.get("lin_vel",1.0),
            self.obs_scales.get("lin_vel",1.0),
            self.obs_scales.get("ang_vel",1.0)
        ], device=self.device)
        self.actions = torch.zeros((N, A), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((N,3), device=self.device)
        self.base_quat = torch.zeros((N,4), device=self.device)
        self.base_euler = torch.zeros((N,3), device=self.device)
        self.current_ankle_heights = torch.zeros((N, 2), device=self.device)
        self.prev_contact = torch.zeros((N, 2), device=self.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][j] for j in self.env_cfg["joint_names"]],
            device=self.device)
        self.extras = {"observations": {}}

    def _resample_commands(self, env_ids):
        # env_ids: Tensor mit Indizes der Envs, die gerade resampled werden sollen
        low, high = self.command_cfg["command_ranges"]["lin_vel_x"]
        self.commands[env_ids,0] = gs_rand_float(low, high, (len(env_ids),), self.device)
        low, high = self.command_cfg["command_ranges"]["lin_vel_y"]
        self.commands[env_ids,1] = gs_rand_float(low, high, (len(env_ids),), self.device)
        low, high = self.command_cfg["command_ranges"]["ang_vel_yaw"]
        self.commands[env_ids,2] = gs_rand_float(low, high, (len(env_ids),), self.device)

    def reset_idx(self, env_ids):
        # Full‐Reset statt Teil‐Reset (fixes len() of a 0‑d tensor)
        self.scene.reset()
        # Buffers für diese Envs zurücksetzen
        self.reset_buf[env_ids]          = 0
        self.episode_length_buf[env_ids] = 0
        for name in self.episode_sums:
            self.episode_sums[name][env_ids] = 0.0
        # Commands für diese Envs neu ziehen
        self._resample_commands(env_ids)
        # Obs‑Buffer für diese Envs initialisieren
        self.obs_buf[env_ids] = 0.0

    def reset(self):
        self.reset_buf[:] = 1
        self.episode_length_buf[:] = 0
        for key in self.episode_sums:
            self.episode_sums[key].zero_()
        self.scene.reset()
        # Befehle für alle Envs neu ziehen
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_commands(all_ids)
        return self.obs_buf, None


    def step(self, actions):
        # 1) Actions speichern und anwenden
        self.last_actions[:] = self.actions
        self.actions = torch.clip(
            actions,
            -self.env_cfg.get("clip_actions", 100.0),
            self.env_cfg.get("clip_actions", 100.0)
        )
        target = self.actions * self.env_cfg.get("action_scale", 1.0) + self.default_dof_pos
        self.robot.control_dofs_position(target, self.motors_dof_idx)

        # 2) Simulationsschritt
        self.scene.step()

        # 3) Zustände updaten
        self.last_torques = torch.zeros_like(self.dof_pos)
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_q)
        self.base_euler[:] = quat_to_xyz(self.base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position()[..., self.motors_dof_idx]
        self.dof_vel[:] = self.robot.get_dofs_velocity()[..., self.motors_dof_idx]
        self.current_ankle_heights[:] = torch.stack(
            [link.get_pos()[:, 2] for link in self.ankle_links],
            dim=1
        )

        # 4) Abbruchkriterien
        done = self.episode_length_buf >= self.max_episode_length
        done |= torch.abs(self.base_euler[:, 1]*math.pi/180) > self.reward_cfg["pitch_threshold"]
        done |= torch.abs(self.base_euler[:, 0]*math.pi/180) > self.reward_cfg["roll_threshold"]
        self.reset_buf = done  # <-- jetzt steht reset_buf korrekt

        # 5) Rewards berechnen (jetzt mit korrekter reset_buf)
        self.rew_buf[:] = 0
        per_step = {}
        for name, fn in self.reward_functions.items():
            r = fn() * self.reward_scales[name]
            self.rew_buf += r
            self.episode_sums[name] += r
            per_step[name] = r

        # 6) Beobachtungen & Extras wie gehabt
        obs_buf, obs_extras = self.get_observations()
        self.extras = {
            "observations": obs_extras["observations"],
            "episode": per_step,
        }

        # 7) Extras mit critic‑Obs und episode‑Rewards befüllen
        self.extras = {
            "observations": obs_extras["observations"],
            "episode": per_step,
        }

        # 8) Episodenlänge inkrementieren
        self.episode_length_buf += 1

        # 9) Kommandos bei Bedarf neu sampeln
        resample_every = int(self.command_cfg["resampling_time_s"] / self.dt)
        idx = (self.episode_length_buf % resample_every == 0).nonzero(as_tuple=False).flatten()
        if idx.numel() > 0:
            self._resample_commands(idx)

        # 10) Tatsächliches Zurücksetzen der beendeten Envs
        done_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.reset_idx(done_ids)

        # 11) (Optional) Debug-Ausgabe am Episodenanfang
        if self.episode_length_buf[0] == 1:
            print("[DEBUG] Observation (env 0):", self.obs_buf[0])
            print("[DEBUG] Action      (env 0):", actions[0])
            print("[DEBUG] Reward      (env 0):", self.rew_buf[0])

        # 12) Ergebnis zurückgeben
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras



    def get_observations(self):
        base_lin_vel = self.base_lin_vel * self.obs_scales.get("lin_vel", 1.0)
        base_ang_vel = self.base_ang_vel * self.obs_scales.get("ang_vel", 1.0)
        dof_pos = self.dof_pos * self.obs_scales.get("dof_pos", 1.0)
        dof_vel = self.dof_vel * self.obs_scales.get("dof_vel", 1.0)
        joint_torques = self.last_torques
        commands = self.commands

        obs = torch.cat((base_lin_vel, base_ang_vel, dof_pos, dof_vel, joint_torques, commands), dim=-1)
        self.obs_buf[:] = obs
        return self.obs_buf, {"observations": {"critic": obs.clone()}}

    def get_privileged_observations(self):
        return None
   


    # ---------------------------------------------------
    # Reward Funktionen (überarbeitet)
    # ---------------------------------------------------

    @register_reward()
    def _reward_periodic_gait(self):
        """
        Phasenorientierte Gait‑Shaping‑Belohnung:
        Linke Stance‑Hälfte, rechte Swing‑Hälfte, dann umgekehrt.
        """
        phase = (self.episode_length_buf.float() * self.dt) % self.reward_cfg["period"]
        half = self.reward_cfg["period"] * 0.5
        contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
        desired_left = (phase < half).float()
        desired_right = (phase >= half).float()
        # positiv im Bereich [0,1]
        return desired_left * contact[:, 0] + desired_right * contact[:, 1]


    @register_reward()
    def _reward_energy_penalty(self):
        """
        Energie‑Effizienz als Gauß‑Reward statt -Summe.
        Minimierung der Aktionsänderungen.
        """
        err = torch.sum((self.actions - self.last_actions)**2, dim=1)
        sigma = self.reward_cfg.get("energy_sigma", 1.0)
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_foot_swing_clearance(self):
        """
        Belohnung für Fuß‑Höhenüberschuss im Swing.
        """
        hs = self.current_ankle_heights
        contact = (hs < self.CONTACT_HEIGHT).float()
        swing_mask = 1.0 - contact
        clearance = hs * swing_mask
        excess = torch.relu(clearance - self.reward_cfg["clearance_target"])
        return torch.mean(excess, dim=1)


    @register_reward()
    def _reward_forward_torso_pitch(self):
        """
        Gauß‑Reward für Vorwärts‑Pitch nahe einem Sollwert.
        """
        pitch = self.base_euler[:, 1] * math.pi / 180
        err = (pitch - self.reward_cfg["pitch_target"])**2
        sigma = self.reward_cfg["pitch_sigma"]
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_knee_extension_at_push(self):
        """
        Belohnt gestrecktes Knie in Standphase (Kontakt).
        """
        hs = self.current_ankle_heights
        stance = (hs < self.CONTACT_HEIGHT).any(dim=1).float()
        idx_l = self.env_cfg["joint_names"].index("Left_KNEE_FE")
        idx_r = self.env_cfg["joint_names"].index("Right_SHIN_FE")
        ext_l = 1.0 - torch.relu(self.dof_pos[:, idx_l])
        ext_r = 1.0 - torch.relu(-self.dof_pos[:, idx_r])
        return stance * ((ext_l + ext_r) * 0.5)


    @register_reward()
    def _reward_tracking_lin_vel(self):
        """
        Gauß‑Reward für lin. Geschw‑Tracking in x/y.
        """
        err = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2])**2, dim=1)
        sigma = self.reward_cfg["tracking_sigma"]
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_tracking_ang_vel(self):
        """
        Gauß‑Reward für ang. Geschw‑Tracking in z.
        """
        err = (self.commands[:, 2] - self.base_ang_vel[:, 2])**2
        sigma = self.reward_cfg["tracking_sigma"]
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_orientation_stability(self):
        """
        Gauß‑Reward für kleine Roll‑/Pitch‑Abweichung.
        """
        roll = self.base_euler[:, 0] * math.pi / 180
        pitch = self.base_euler[:, 1] * math.pi / 180
        err = roll**2 + pitch**2
        sigma = self.reward_cfg.get("orient_sigma", 0.1)
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_base_height(self):
        """
        Gauß‑Reward für Hüfthöhe nahe Ziel.
        """
        err = (self.base_pos[:, 2] - self.reward_cfg["base_height_target"])**2
        sigma = self.reward_cfg.get("height_sigma", 0.1)
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_survive(self):
        """
        Belohnt 1.0, solange die Env NICHT im Fallen‑Zustand ist,
        und 0 beim ersten Überschreiten der Winkel‑Limits.
        """
        # self.reset_buf wird kurz darauf gesetzt:
        done = self.reset_buf.float()  # 1.0, sobald Episode vorbei (fallen oder Länge)
        return (1.0 - done)

    
    @register_reward()
    def _reward_fall_penalty(self):
        """
        Bestrafe Umfallen: sobald Roll oder Pitch über den Schwellwert gehen.
        Gibt –1.0 pro Step zurück, wenn überschritten.
        """
        # Roll und Pitch in Radiant
        roll  = self.base_euler[:, 0] * math.pi / 180
        pitch = self.base_euler[:, 1] * math.pi / 180

        # Thresholds aus reward_cfg (z.B. 30° in Radiant)
        thr_r = self.reward_cfg["roll_threshold"]
        thr_p = self.reward_cfg["pitch_threshold"]

        # Maske, wo einer der Winkel überschritten ist
        mask = ((roll.abs() > thr_r) | (pitch.abs() > thr_p)).float()

        # Als Penalty skaliert –1 pro Step
        return -mask



    @register_reward()
    def _reward_bird_hip_phase(self):
        """
        Vogel‑typischer Hüft‑FE‑Zyklustreiber als Gauß‑Reward.
        """
        idx_l = self.env_cfg["joint_names"].index("Left_THIGH_FE")
        idx_r = self.env_cfg["joint_names"].index("Right_THIGH_FE")
        phase = ((self.episode_length_buf.float() * self.dt) % self.reward_cfg["period"]) / self.reward_cfg["period"]
        omega = 2 * math.pi * phase
        tgt  = self.reward_cfg["bird_hip_target"]
        amp  = self.reward_cfg["bird_hip_amp"]
        desired_l = tgt + amp * torch.sin(omega)
        desired_r = tgt - amp * torch.sin(omega)
        a_l = self.dof_pos[:, idx_l]
        a_r = self.dof_pos[:, idx_r]
        err = (a_l - desired_l)**2 + (a_r - desired_r)**2
        sigma = self.reward_cfg["bird_hip_sigma"]
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_hip_abduction_penalty(self):
        """
        Gauß‑Strafe für Hüft‑AA Abduktion/Adduktion.
        """
        idx_l = self.env_cfg["joint_names"].index("Left_HIP_AA")
        idx_r = self.env_cfg["joint_names"].index("Right_HIP_AA")
        abd_l = self.dof_pos[:, idx_l]
        abd_r = self.dof_pos[:, idx_r]
        err = abd_l**2 + abd_r**2
        sigma = self.reward_cfg["hip_abduction_sigma"]
        return torch.exp(-err / (2 * sigma**2))


    @register_reward()
    def _reward_lateral_drift_penalty(self):
        """
        Gauß‑Reward für geringe seitliche Drift (y‑Geschw.).
        """
        drift = self.base_lin_vel[:, 1].abs()
        sigma = self.reward_cfg.get("drift_sigma", 0.1)
        return torch.exp(-drift**2 / (2 * sigma**2))
























































    # Old Reward Functions

    @register_reward()
    def _reward_lin_vel_z(self):
        return self.base_lin_vel[:, 2]**2

    @register_reward()
    def _reward_action_rate(self):
        return torch.sum((self.last_actions - self.actions)**2, dim=1)

    @register_reward()
    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    @register_reward()
    def _reward_penalize_hip_fe(self):
        idx = self.env_cfg["joint_names"].index("Left_THIGH_FE")
        return torch.abs(self.dof_pos[:, idx])

    @register_reward()
    def _reward_penalize_hip_fe_diff(self):
        i0 = self.env_cfg["joint_names"].index("Left_THIGH_FE")
        i1 = self.env_cfg["joint_names"].index("Right_THIGH_FE")
        return torch.abs(self.dof_pos[:, i0] - self.dof_pos[:, i1])

    @register_reward()
    def _reward_penalize_knee_fe_left(self):
        i0 = self.env_cfg["joint_names"].index("Left_KNEE_FE")
        return torch.relu(0.9 + self.dof_pos[:, i0])

    @register_reward()
    def _reward_penalize_knee_fe_right(self):
        i1 = self.env_cfg["joint_names"].index("Right_SHIN_FE")
        return torch.relu(-self.dof_pos[:, i1] + 0.9)

    @register_reward()
    def _reward_penalize_ankle_height(self):
        return torch.mean(self.current_ankle_heights, dim=1)

    @register_reward()
    def _reward_gait_regularity(self):
        left = self.dof_pos[:, self.env_cfg["joint_names"].index("Left_THIGH_FE")]
        right = self.dof_pos[:, self.env_cfg["joint_names"].index("Right_THIGH_FE")]
        phase_diff = torch.abs(left + right)
        return torch.exp(-phase_diff / 0.3)

    @register_reward()
    def _reward_foot_orientation(self):
        return torch.zeros(self.num_envs, device=self.device)

    @register_reward()
    def _reward_step_height_consistency(self):
        left = self.current_ankle_heights[:, 0]
        right = self.current_ankle_heights[:, 1]
        diff = torch.abs(left - right)
        return torch.exp(-diff / 0.05)

    @register_reward()
    def _reward_foot_contact_penalty(self):
        h = self.current_ankle_heights
        contact = (h < self.CONTACT_HEIGHT).float()
        flight = (contact.sum(dim=1) == 0).float()
        one = (contact.sum(dim=1) == 1).float()
        max_h = torch.max(h, dim=1)[0]
        hop = one * torch.relu(max_h - self.SWING_HEIGHT_THRESHOLD)
        return flight + hop

    @register_reward()
    def _reward_foot_contact_switch(self):
        h = self.current_ankle_heights
        contact = (h < self.CONTACT_HEIGHT).float()
        change = (contact != self.prev_contact).float()
        both = change[:,0] * change[:,1]
        self.prev_contact[:] = contact
        return both

    @register_reward()
    def _reward_posture_stability(self):
        roll = self.base_euler[:,0] * math.pi / 180
        pitch = self.base_euler[:,1] * math.pi / 180
        return torch.exp(-(roll**2 + pitch**2) / 0.05)

    @register_reward()
    def _reward_smooth_actions(self):
        return -torch.sum((self.actions - self.last_actions)**2, dim=1)

    @register_reward()
    def _reward_leg_swing_forward(self):
        l = self.dof_pos[:, self.env_cfg["joint_names"].index("Left_THIGH_FE")]
        r = self.dof_pos[:, self.env_cfg["joint_names"].index("Right_THIGH_FE")]
        return torch.exp(-((l - (-r))**2) / 0.1)

    @register_reward()
    def _reward_yaw_stability(self):
        return self.base_ang_vel[:,2]**2

    @register_reward()
    def _reward_gait_phase(self):
        phase = (self.episode_length_buf.float() % 100) / 100 * 2 * math.pi
        lt = 0.3 * torch.sin(phase)
        rt = -0.3 * torch.sin(phase)
        l = self.dof_pos[:, self.env_cfg["joint_names"].index("Left_THIGH_FE")]
        r = self.dof_pos[:, self.env_cfg["joint_names"].index("Right_THIGH_FE")]
        err = (l - lt)**2 + (r - rt)**2
        return torch.exp(-err / 0.02)

    @register_reward()
    def _reward_gait_phased_leg_control(self):
        return torch.zeros(self.num_envs, device=self.device)

    @register_reward()
    def _reward_knee_hyperextension(self):
        kp = self.dof_pos[:, self.env_cfg["joint_names"].index("Left_KNEE_FE")]
        kp2 = self.dof_pos[:, self.env_cfg["joint_names"].index("Right_SHIN_FE")]
        return torch.relu(kp)**2 + torch.relu(kp2)**2

    @register_reward()
    def _reward_flat_feet(self):
        return torch.zeros(self.num_envs, device=self.device)

    @register_reward()
    def _reward_bird_knee_posture(self):
        lt = self.dof_pos[:, self.env_cfg["joint_names"].index("Left_THIGH_FE")]
        rt = self.dof_pos[:, self.env_cfg["joint_names"].index("Right_THIGH_FE")]
        lk = self.dof_pos[:, self.env_cfg["joint_names"].index("Left_KNEE_FE")]
        rk = self.dof_pos[:, self.env_cfg["joint_names"].index("Right_SHIN_FE")]
        pr = -((lk - lt)**2 + (rk - rt)**2)
        pen = 10 * (torch.relu(lk)**2 + torch.relu(rk)**2)
        return pr - pen

    @register_reward()
    def _reward_foot_contact_phase_based(self):
        cf = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
        return torch.abs(cf[:,0] - cf[:,1])

