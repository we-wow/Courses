import os

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
from DeepQNetwork import DQN
import numpy as np
import datetime
import tensorflow as tf


MAX_EPOCHS = 300

env = gym.make('CartPole-v0')
env = env.unwrapped
agent = DQN(
    s_dim=env.observation_space.shape[0],
    a_dim=env.action_space.n,
    gamma=0.9,
    batch_size=32,
    lr=0.001,
    replace_target_iter=50,
    memory_capacity=1000,
    e_greedy_increment=None
)


def my_reward(observations):
    x, _, theta, _ = observations  # observations = [x,x导数,theta,theta导数]
    rp = (env.x_threshold - abs(x))/env.x_threshold - 0.8  # reward position
    ra = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5  # reward angle
    return rp + ra


version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'log/' + version
summary_writer = tf.summary.create_file_writer(log_dir)

loss = 0
rewards = []
for epoch in range(MAX_EPOCHS):
    s = env.reset()
    reward = 0
    for i in range(200):
        env.render()
        a = agent.get_action(s)
        s_, r, done, info = env.step(a)
        r = my_reward(s_)
        if done and i < 199:
            r = -10
        reward += r
        agent.store_transition(s, a, r, s_, done)
        if agent.pointer > agent.memory_capacity:
            loss = agent.learn()
        if done:
            break
        s = s_

    evaluation_reward = 0
    s = env.reset()
    for i in range(200):
        env.render()
        a = np.argmax(agent.model(np.array([s], dtype=np.float32))[0])
        s_, r, done, info = env.step(a)
        evaluation_reward += r
        if done:
            break
        s = s_
    rewards.append(evaluation_reward)
    epsilon_increased = agent.epsilon * 1.02
    if epsilon_increased <= 0.9:
        agent.epsilon = epsilon_increased
    else:
        agent.epsilon = 0.9
    print(f"Epoch :{epoch:04d} Reward: {reward:5.1f} Eva: {evaluation_reward:5.1f} Epsilon: {agent.epsilon:.2f} "
          f"Pointer: {agent.pointer}")
    with summary_writer.as_default():
        tf.summary.scalar('episode reward', reward, step=epoch)
        tf.summary.scalar('Eva', evaluation_reward, step=epoch)
        tf.summary.scalar('loss', loss, step=epoch)

agent.model.save(f"model/model_{version}.h5")
plt.plot(range(len(rewards)), rewards)