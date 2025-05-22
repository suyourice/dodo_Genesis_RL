# dodo_Genesis_RL
TUM MIRMI AIM Dodo Alive! Project \
# How to launch tensorboard
tensorboard --logdir logs/dodo-walking 
# Where to put the robot model file
/Genesis/genesis/assets/urdf/dodo_robot$ \
make sure xml file and .obj in same layer
# Where put all source files (train, eval, env)
/Genesis/examples/locomotion$ 
# Sanity Check and Start with
First run go2 example as in documentation\
Then try to run import robot to see dodo in Genesis
# Reward implementation workflow
In Env: Extract data in simulation and store it in variable, then calculate reward value\
In Train: Register the reward term and assign the weight\
In import_robot: Doing sanity check if needed
