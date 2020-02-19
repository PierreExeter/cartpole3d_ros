import gym
import numpy as np
import pybulletgym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import time

env = gym.make('ReacherPyBulletEnv-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model_list = [
        A2C(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs/A2C/"), 
        ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs/ACKTR/"), 
        DDPG(mlp_ddpg, env, verbose=1, tensorboard_log="tensorboard_logs/DDPG/"),
        PPO1(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs/PPO1/"),
        PPO2(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs/PPO2/"), 
        SAC(mlp_sac, env, verbose=1, tensorboard_log="tensorboard_logs/SAC/"), 
        TRPO(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs/TRPO/"),
        TD3(mlp_td3, env, action_noise=action_noise, verbose=1, tensorboard_log="tensorboard_logs/TD3/"),
]

algo_list = ['A2C', 'ACKTR', 'DDPG', 'PPO1', 'PPO2', 'SAC', 'TRPO', 'TD3']


model_list = [model_list[5], model_list[5]]
algo_list = [algo_list[5], algo_list[5]]

env.render(mode="human")  # this needs to be placed BEFORE env.reset() 


for model, algo in zip(model_list, algo_list):
    print(algo)
    model = model.load("trained_models/"+algo)
    obs = env.reset()

    reward_list = []

            
    for i in range(500):
        if i % 100 == 0:
            print(i)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)

            

env.close()
