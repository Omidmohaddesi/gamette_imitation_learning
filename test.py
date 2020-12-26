import numpy as np
import os
import csv
import gym
import gym_crisp
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
from gym.wrappers.filter_observation import FilterObservation

def callback(locals_, globals_):
    self_ = locals_['self']
    seg_ = locals_['seg_gen'].__next__()
    true_rewrds_ = seg_['true_rewards']
    mask_ = seg_['dones']
    value = sum(true_rewrds_) / sum(mask_)
    # value = self_.episode_reward
    summary = tf.Summary(value=[tf.Summary.Value(tag='Average_Episodes_Reward', simple_value=value)])
    locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


if __name__ == '__main__':

    # Create and wrap the environment
    # env = gym.make('Crisp-v2', study_name='study_2_3', start_cycle=60)
    env = FilterObservation(gym.make('Crisp-v2', study_name='study_2_3', start_cycle=60),
                            filter_keys=['inventory', 'demand-hc1', 'demand-hc2', 'on-order',
                                         'shipment', 'suggestion', 'outl'])
    prob = []
    action_list = []
    reward_list = []
    reward_sum = 0
    obs = env.reset()
    n_steps = 36
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        action_list.append(action)
        # print('Reward is: ', reward)
        print(info['time'])
        if done:
            print(f'Total reward is {reward_sum} \n')
            reward_list.append(reward_sum)
            reward_sum = 0
            obs = env.reset()

    print('Average reward is: ', int(np.mean(reward_list)))
    print('SD of reward is: ', int(np.std(reward_list)))
