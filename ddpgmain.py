import argparse
import os,sys
import time

#import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
import tensorlayer as tl
import matlab.engine
import numpy as np

#####################  hyper parameters  ####################

ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 123  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'DDPG'
TRAIN_EPISODES = 100  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 200  # total number of steps for each episode

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 32  # update action batch size
VAR = 4  # control exploration

###############################  DDPG  ####################################
#預設路徑 200筆
plan_dep = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-0.0022,-0.0122,-0.0309,-0.0572,-0.0900,-0.1284,-0.1713,-0.2179,-0.2674,-0.3189,-0.3716,-0.4250,-0.4784,-0.5311,-0.5826,-0.6324,-0.6801,-0.7253,-0.7676,-0.8068,-0.8427,-0.8749,-0.9035,-0.9283,-0.9493,-0.9665,-0.9799,-0.9897,-0.9961,-0.9993,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000,-1.0000]#env
plan_pitch =[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   -0.6831,   -1.5436,   -2.5641,   -3.7246,   -5.0052,   -6.3859,   -7.8466,   -9.3673,  -10.9281,  -12.5090,  -14.0900,  -15.6509,  -17.1720,  -18.6331,  -20.0143,  -18.7200,  -17.4800,  -16.2800,  -15.1200,  -14.0000,  -12.9200,  -11.8800,  -10.8800,   -9.9200,   -9.0000,   -8.1200,   -7.2800,   -6.4800,   -5.7200,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000,   -5.0000]
class AUVEnvironment(object):
    def __init__(self):
        self.StateVec=[]
        self.PressureSensor=[]
        self.eng=matlab.engine.start_matlab()

    def step(self,init,stern,sec):

        [state_,pressure]=self.eng.step(matlab.double([init.tolist()]),matlab.double([stern.tolist()]),matlab.double([sec]),nargout=2)
        state_=np.array(state_).flatten()
        self.StateVec.append(state_.tolist())

        pressure=np.array(pressure)
        self.PressureSensor.append(pressure.flatten().tolist())
        
        #reward
        dep_error=abs(plan_dep[sec]-(-1*pressure)) # 深度誤差絕對值
        pitch_error=abs(plan_pitch[sec]-(state_[10])*57.3) # pitch angle 誤差絕對值
        reward=0.6*(-1)*dep_error+0.4*(-1)*pitch_error*1.05/180
        
        #done
        if sec>MAX_STEPS:
            done=1
        #elif self.static_times>=5:
            #done=1
        else:
            done=0

        return state_, reward, done

    def reset(self):
        #清空狀態矩陣
        self.StateVec=[]
        self.PressureSensor=[]
        init = [0.5 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
        self.StateVec.append(init)
        
        return np.array(init)

class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, action_dim, state_dim, action_range):
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)#創建初始化記憶體
        self.pointer = 0
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR
        #初始化權重
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            input_layer = tl.layers.Input(input_state_shape, name='A_input')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer) #超過範圍的action可能導致程序異常 使用tanh將輸出map to -1~1
            layer = tl.layers.Lambda(lambda x: action_range * x)(layer)#再用lamda map to 環境的取值範圍
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            #在DQN會把某state的所有action Q值列出 選擇max q action 輸出 在DDPG也是 由net 去判斷q值 故需要將act跟state輸入critic net
            state_input = tl.layers.Input(input_state_shape, name='C_s_input')
            action_input = tl.layers.Input(input_action_shape, name='C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])#將兩個輸入層做結合
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(layer)
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])#建構actor net
        self.critic = get_critic([None, state_dim], [None, action_dim])#建構critic net 
        self.actor.train()
        self.critic.train()

        #更新參數 用於首次賦值
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)
        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, state_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()
        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement 建立滑動平均值 TAU表示加入的新元素的比例 1-TAU表示舊平暈值的比例

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)
    #滑動平均值更新
    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        #先確定要更新的變量 並使用apply添加相應舊均值副本
        paras = self.actor.trainable_weights + self.critic.trainable_weights #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)
        #將變量對應並更新 但在更新前先計算滑動平均值再賦值到target
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))# 用滑动平均赋值

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        return np.clip(
            np.random.normal(a, self.var), -self.action_range, self.action_range
        )  # add randomness to action selection for exploration
    #net 參數更新
    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE) 
        datas = self.memory[indices, :]#隨機抽取數據
        states = datas[:, :self.state_dim]#取出狀態
        actions = datas[:, self.state_dim:self.state_dim + self.action_dim]#取出動作
        rewards = datas[:, -self.state_dim - 1:-self.state_dim]#取出reward
        states_ = datas[:, -self.state_dim:]#取出執行a得到的狀態
           
        #critic更新
        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)#將新狀態輸入actor target得到 action target
            q_ = self.critic_target([states_, actions_])#輸入critic target 得到q target
            y = rewards + GAMMA * q_ #更新目標y
            q = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)#計算目標y跟預測q的均方誤差
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
        #actor更新
        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q)  # maximize the q 更新方式為梯度上升 故前面要加上負號
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()#滑動平均更新(將新的參數賦值給target net)
    
    
    
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_)) #將資料水平堆疊
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

#儲存資料
def createdatafold():
    import datetime
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    fpath=f"./data/{mkfile_time}"
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
    
    print("Environment,matlab engine done!!")
    #env = gym.make(ENV_ID).unwrapped
    
    # reproducible 隨機種子
    #env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    #state_dim = env.observation_space.shape[0] #為3
    state_dim = 12
    #action_dim = env.action_space.shape[0] #為1
    action_dim = 1
    #action_range = env.action_space.high  # scale action, [-action_range, action_range] [2.]
    action_range = np.array([25.])#舵板角度上限
    #print(action_range,type(action_range))#[30.] <class 'numpy.ndarray'>
    agent = DDPG(action_dim, state_dim, action_range)
    print("Agent create finished!!")
    fpath=createdatafold()
    print("Folder finished!!")
    # train
    all_episode_reward = []
    recent_episode_reward = np.zeros(5)
    recenttruefalse = 0
    i=1
    try:
        print("Start training ...")
        #agent.load()#接續上一次的
        for episode in range(TRAIN_EPISODES):
            t0 = time.time()
            state = env.reset()
            stern_angle=[]
            episode_reward = 0
            for step in range(MAX_STEPS):
                #if step>=20:
                action = agent.get_action(state)#得到動作 action>0代表向下 action<0代表向上
                #else:
                    #action = np.array(0)
                stern_angle.append(action.flatten().tolist())
                state_, reward, done = env.step(state,action,step)

                agent.store_transition(state, action, reward, state_)#儲存歷史資料

                if agent.pointer > MEMORY_CAPACITY: #如果數據量足夠，便隨機抽樣，更新actor和critic 
                    agent.learn()
                    recenttruefalse = 1
                state = state_ #更新狀態值
                episode_reward += reward
                if done:
                    break
            savecsv(i,fpath)
            i+=1
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)#old reward*0.9 new reward*0.1
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
            recent_episode_reward[episode%5]=episode_reward
            if recenttruefalse == 1:
                if sum(recent_episode_reward)/5 >= -1:
                    print("Error > -1! Stop Training...")
                    break
        agent.save()
        print("Model saved!!")
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
        print("Training is done!!")


    except KeyboardInterrupt:
        try:
            sys.exit(0) #引發一個異常 退出編譯器 若在子執行緒使用就只能退出子執行緒 主執行緒仍然能運作
        except SystemExit:
            os._exit(0) #直接將程式終止