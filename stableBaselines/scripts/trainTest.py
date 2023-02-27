import gym
#import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import os 
import yaml
from yaml.loader import SafeLoader
from sb3_contrib import TQC
import time
import numpy as np
import torch as th
###########################################################################
### This script trains the CartPole environment with PPO RL algorithm
### and saves the model and model history. It has 2 functionality, Training
### and inference...
### USAGE: 
# For training python3 cartPolePPOTrainTest.py -envName "CartPole-v1" -outputName "cartPolePPO" -mode "1"
# For testing python3 cartPolePPOTrainTest.py -envName "CartPole-v1" -outputName "cartPolePPO" -mode "0"
# To see the tensorboard , go to tensorboard dir and run: tensorboard --logdir .
###########################################################################

def train(parentDir,config, algorithm,env):
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='/home/tumu/anaconda3/envs/stableBaselines/model',
                                         name_prefix='PandaReach-v2TQC_'+config['expNumber'])
    modelDir = os.path.join(parentDir, "model")
    logDir = os.path.join(parentDir, "log")
    if config['algorithm']=="PPO":
        #policy_kwargs= dict(
         #           log_std_init=-2,
          #          ortho_init=False,
           #         activation_fn=th.nn.ReLU,
            #        net_arch=dict(pi=[128, 128], vf=[128, 128])
             #     )
        policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[128, 128,128], vf=[128,128, 128])])
        model = algorithm(policy=config['policy'], env=env, n_steps=config['n_steps'],verbose=1, batch_size=config['batch_size'], learning_rate=config['learning_rate'],
                          tensorboard_log=logDir+"/"+config['envName']+"_"+config['algorithm']+"_"+config['expNumber'],policy_kwargs=policy_kwargs)
    
    if config['algorithm']=="DDPG":
        model = algorithm(policy=config['policy'], env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
             gamma=config['gamma'], batch_size=config['batch_size'], buffer_size=config['buffer_size'], learning_rate = 1e-3, replay_buffer_kwargs = config['rb_kwargs'],
             policy_kwargs = config['policy_kwargs'], tensorboard_log=logDir+"/"+config['envName']+""+config['algorithm'])
    
    if config['algorithm']=="TQC":
        #policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[128, 128,128], vf=[128,128, 128])])
        #policy_kwargs = dict(activation_fn=th.nn.Tanh,net_arch=dict(pi=[128, 128], qf=[128, 128]))
        if config['continueTraining'] ==True:
            #model = algorithm.load(modelDir+"/"+config['envName']+""+config['algorithm']+"_"+config['expNumber'], env=env)
            #model.set_env(env)
            pass
        else:
            model = algorithm(policy=config['policy'], env=env, tensorboard_log=logDir+"/"+config['envName']+""+config['algorithm'],
                            verbose=1, ent_coef=config['ent_coef'], batch_size=config['batch_size'], gamma=config['gamma'],
                            learning_rate=config['learning_rate'], learning_starts=config['learning_starts'],replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=config['replay_buffer_kwargs'], policy_kwargs=config['policy_kwargs'])
    model.learn(total_timesteps=config['total_timesteps'], callback=checkpoint_callback)
    model.save(modelDir+"/"+config['envName']+""+config['algorithm']+"_"+config['expNumber'])     

    del model

def load_model(parentDir,config,steps, algorithm,env):
    modelDir = os.path.join(parentDir, "model")
    model = algorithm.load(modelDir+"/"+config['envName']+""+config['algorithm']+"_"+config['expNumber'], env=env)
    
    env = model.get_env()
    for step in range(steps):
        squaredError = 0
        done = False
        obs = env.reset()
        episode_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            #print("action in  traintest:", action)
            obs, reward, done, info = env.step(action)
            #print("obs in traintest:", obs)
            #print("reward:", reward)
            if info[0]["is_success"]==1:
                error = abs(obs['achieved_goal'] - obs['desired_goal'])
                squaredError += np.sum(error**2)
            if done:
                #print("info: ",info)
                pass

            time.sleep(0.01)
            episode_reward+=reward
            #env.render()
        print("episode reward is:", episode_reward)
    print("Squared Error:",squaredError)
    print("RMSE:", np.sqrt((squaredError)/(steps*50)))
        
        

def main():
    with open('configTQC.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if config['algorithm']=="PPO":
        algorithm = PPO
        env = gym.make(config['envName'], render=config['render'])
    if config['algorithm']=="DDPG":
        algorithm = DDPG
        env = gym.make(config['envName'], render=config['render'])
    if config['algorithm']=="TQC":
        algorithm = TQC
        env = gym.make(config['envName'], render=config['render'])
        #env = gym.make(config['envName'])
        

    
    if config['mode'] == True:
        train(parentDir, config,algorithm, env)
    else:
        load_model(parentDir,config,20, algorithm,env)
    
main()
