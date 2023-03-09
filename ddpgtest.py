import argparse
import os,sys
import time
import traceback  
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl
import matlab.engine
import numpy as np
from ddpgmain import AUVEnvironment,DDPG,plan_dep
#####################  hyper parameters  ####################

#ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'DDPG'
TRAIN_EPISODES = 500  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 200  # total number of steps for each episode

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 32  # update action batch size
VAR = 4  # control exploration

#儲存資料
def createdir():
    import datetime
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    fpath=f"./test/{mkfile_time}"
    if not os.path.exists(fpath):  
        os.makedirs(fpath)
    return fpath
def savecsv(i,fpath):
    import csv
    
    with open(f"{fpath}/state{i}.csv","w",newline="") as f:
        cw=csv.writer(f) 
        cw.writerows(env.StateVec[j] for j in range(len(env.StateVec)))
    
    with open(f"{fpath}/PressureSensor{i}.csv","w",newline="") as f:
        cw=csv.writer(f)
        cw.writerows(env.PressureSensor[j] for j in range(len(env.PressureSensor)))
    
    with open(f"{fpath}/stern{i}.csv","w",newline="") as f:
        cw=csv.writer(f)
        cw.writerows(stern_angle[j] for j in range(len(stern_angle)))

if __name__ == '__main__':
    env = AUVEnvironment()
    
    # reproducible 隨機種子
  
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    
    state_dim = 12
    
    action_dim = 1
    action_range = np.array([25.])#舵板角度上限
    agent = DDPG(action_dim, state_dim, action_range)

    fpath=createdir()
    i=1

    all_episode_reward = []
    try:
        #test
        agent.load()
        for episode in range(TEST_EPISODES):
            t0 = time.time()
            state = env.reset()
            episode_reward = 0
            episode_error = 0
            stern_angle=[]
            for step in range(MAX_STEPS):
                action = np.array(agent.get_action(state ,greedy=True))
                state_, reward, done = env.step(state,action,step)
                stern_angle.append(action.flatten().tolist())
                episode_reward += reward[0]
                episode_error += reward[1]
                state = state_
                if done:
                    break
            savecsv(i,fpath)
            i+=1
            
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}  | Trajectory Error: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0,episode_error/MAX_STEPS
                )
            )
    except Exception as e:
        print(e)
        traceback.print_exc()
        
        os._exit(0) #直接將程式終止