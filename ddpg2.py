"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def getPath(routeArgs，network，networkState，networkEmbedding,epsilon=0.1): #epsilon : control exploration
    '''
        networkState: 为网络状态
    '''

        s= #s为一个状态，也即将routeArgs中的 currentPosition变为节点embedding的表示，同时需要考虑 networkState

        originExpress=routeArgs[-1]

        currentPosition=originExpress #
        neighborNode=【】 # currentPosition 的邻居节点，这个列表是节点的编号

        if(neighborNode包含目的节点)
           return （1，目的节点编号）

        if np.random.rand() > epsilon: # add randomness to action selection for exploration
            # choose best action
            action = ddpg.choose_action(s)
            return (0,action) #0 表示一个具体的网络embedding数值

        else:
            # choose random action
            originExpress = random.sample(neighborNode，1) # 从currentPosition的邻居节点 随机选择一个
            retrun (1,originExpress)

        #如果选择的点有环路，则环境会返回来一个reward，reward的值，应该很小，表示不想出现环路
        2：如果即将加入的originExpress的所有邻居，已经都在path中，则需要在path中删除后三个，重新设置 originExpress


###############################  training  ####################################







s_dim = 41
a_dim = 30
a_bound=1

ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg.train(routeTuple) # 路径元祖，是一个list：【（vector1，vector2）（vector1，vector2）】，vector是Embedding之后的表示



t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()  #环境初始化
    ep_reward = 0
    for j in range(MAX_EP_STEPS):

        #
        flag，a = ddpg.getpath(routeArgs，network，networkEmbedding)

        if flag==0： # a为具体的网络表示数值

            #比较点a和 neighborNode节点的距离，以及neighborNode和目的节点的距离，需要折中，返回一个节点
            originExpress=compareNode(a,neighborNode) #compareNode函数返回具体的节点编号
        else：
            originExpress=a
            a=getEmbedding(a) #得到节点a的网络表示

        s_, r = env.step(originExpress) #经过节点a会返回的吞吐量
        #在环境中执行动作，获取吞吐量信息，s_是执行这个动作之后，网络的状态，可以用流量矩阵，压缩成一个多维数组

        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            # var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), )
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - t1)