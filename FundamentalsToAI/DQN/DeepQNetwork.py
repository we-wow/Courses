"""

@Time   : 2022/1/16 14:45
@Author : Wei Mingjiang
@File   : DeepQNetwork.py
@Version: 1.0.0
@Content: First version.
"""
import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, s_dim, a_dim, gamma, memory_capacity,
                 batch_size, lr, replace_target_iter=50, e_greedy_increment=None):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.batch_size = batch_size

        self.optimizer = tf.optimizers.Adam(lr)
        self.model = self._build_model("online")
        self.target = self._build_model("target")
        # Replay buffer
        self.pointer = 0
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + 1 + 1 + 1), dtype=np.float32)

        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter

        self.epsilon = 0.1 if e_greedy_increment is None else e_greedy_increment

        self.gamma = gamma

    def _build_model(self, name, hidden_layer=None):
        if hidden_layer is None:
            hidden_layer = [32, 32]
        activation_fun = tf.nn.tanh
        # constrain = tf.keras.constraints.min_max_norm(-3, 3)
        inputs = tf.keras.layers.Input((self.s_dim,))
        x = inputs
        for lay in hidden_layer:
            x = tf.keras.layers.Dense(lay, activation=activation_fun, kernel_initializer='RandomNormal')(x)
        output = tf.keras.layers.Dense(self.a_dim, activation='linear')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=output, name=name)
        return model

    def store_transition(self, s, a, r, s_, done):
        """
        存储历史记录到经验回放池
        :param s: numpy.ndarray [] shape=(3,) 当前状态
        :param a: numpy.ndarray [] shape=(1,) 动作
        :param r: numpy.float64 奖励
        :param s_: numpy.ndarray [] 下一步状态
        :param done: numpy.ndarray [] 最终状态
        :return:
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        # 把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_, [int(done)]))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        """
        更新参数，训练
        :return:
        """
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)  # 随机选取BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]  # 从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + 1].reshape(self.batch_size)  # 从bt获得数据a
        br = bt[:, -self.s_dim - 2:-self.s_dim - 1].reshape(self.batch_size)  # 从bt获得数据r
        bs_ = bt[:, -self.s_dim - 1: -1]  # 从bt获得数据s'
        bd_ = bt[:, -1:].reshape(self.batch_size)  # 从bt获得数据s'

        a_ = self.target(bs_)
        q_ = np.max(a_, axis=1)
        y = np.where(bd_, br, br + self.gamma * q_)
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(self.model(bs) * tf.one_hot(ba, self.a_dim), axis=1)
            loss = tf.losses.mean_squared_error(y, q)
        c_grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(c_grad, self.model.trainable_weights))

        # Every C steps copy weights.
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.learn_step_counter = 0
            self.target.set_weights(self.model.get_weights())
        return loss

    def get_action(self, states):
        """
        Epsilon 贪心策略
        :param states:
        :return:
        """
        if np.random.random() > self.epsilon:
            return np.random.choice(self.a_dim)
        else:
            return np.argmax(self.model(np.array([states], dtype=np.float32))[0])

