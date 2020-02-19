#!/usr/bin/env python3
import rospy
import gym
import os

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

from openai_ros.task_envs.cartpole_stay_up import stay_up
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment 


os.chdir('/home/pierre/catkin_ws/src/openai_examples/cartpole/cartpole3d/scripts/')

rospy.init_node('cartpole3d_trpo', anonymous=True, log_level=rospy.FATAL)

environment_name = rospy.get_param('/cartpole_v0/task_and_robot_environment_name')
env = StartOpenAI_ROS_Environment(environment_name)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs/TRPO_cartpole/")
print(model)

# TRAIN
# model.learn(total_timesteps=1000)
# model.save("trained_models/TRPO")

# TEST
model = model.load("trained_models/TRPO")

obs = env.reset()
rate = rospy.Rate(30)

for t in range(1000):
    print(t)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    rate.sleep()

env.close()


