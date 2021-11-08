import gym
import numpy as np
import tensorflow.compat.v1 as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers import FlattenObservation
from stable_baselines import PPO2
import os
from tqdm import tqdm
import pdb
import sys
from config import config

import gym_crisp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


crispPath = os.path.join(os.path.abspath('../..'), 'crisp/')
crisp_server_Path = os.path.join(os.path.abspath('../..'), 'crisp_game_server/')
sys.path.insert(1, crispPath)
sys.path.insert(1, crisp_server_Path)


def callback(locals_, globals_):
    self_ = locals_['self']
    value = np.mean(self_.episode_reward)
    # value = self_.episode_reward
    summary = tf.Summary(value=[tf.Summary.Value(tag='Average_Episodes_Reward', simple_value=value)])
    locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


def main():

    # Create and wrap the environment
    if config.n_cpu == 1:
        env = FlattenObservation(
            FilterObservation(gym.make(config.env_name,
                                       study_name=config.study_name,
                                       start_cycle=config.start_cycle,
                                       scaler=None),
                              filter_keys=config.obs_filter_keys))
    else:
        env = SubprocVecEnv([lambda: FlattenObservation(
            FilterObservation(gym.make(config.env_name,
                                       study_name=config.study_name,
                                       start_cycle=config.start_cycle,
                                       scaler=None),
                              filter_keys=config.obs_filter_keys))
                             for _ in range(config.n_cpu)])

    model = PPO2(MlpPolicy, env, verbose=config.verbose,
                 tensorboard_log='./logdir3/',
                 full_tensorboard_log=True)

    if config.load and os.path.exists(config.ckpt_path + '.zip'):
        model = PPO2.load(config.ckpt_path, verbose=config.verbose,
                          tensorboard_log='./logdir3/',
                          full_tensorboard_log=True)
        model.set_env(env)
        print("ckpt loaded !!!!")

    if not config.test:
        model.learn(total_timesteps=config.itr)
        model.save(config.ckpt_path)
        print('ckpt saved')

    else:
        trajs_state = []
        trajs_action = []
        trajs_reward = []

        order_data = np.empty((0, 35), int)
        allocation_data = np.empty((0, 35), int)
        reward_data = np.empty((0, 35), int)

        # aaa = np.load('Hopper-v3_backward_action.npy')

        for _ in tqdm(range(config.sample_size)):
            prestop = False
            traj_states = []
            traj_actions = []
            traj_rewards = []

            orders = np.empty((0, 8), int)
            allocations = np.empty((0, 8), int)

            obs = env.reset()

            for t in range(config.n_step):
                action, _states = model.predict(obs)
                # action = aaa[_][t][np.newaxis, :]

                traj_states.append(obs)
                traj_actions.append(action)

                obs, reward, dones, info = env.step(action)
                # print(obs, rewards, dones, info)
                # env.render()
                traj_rewards.append(reward)
                # traj_rewards.append(info[0]['backward_reward'])

                orders = np.append(orders, [[item['order'] for item in info]], axis=0)
                allocations = np.append(allocations, [[item['allocation'] for item in info]], axis=0)


                # order_data = np.append(order_data, [orders], axis=0)
                # allocation_data = np.append(allocation_data, [allocations], axis=0)

                if True in dones:     # if n_cpu is more than 1
                # if dones:               # if n_cpu is 1
                    prestop = True
                    trajs_state.append(traj_states)
                    trajs_action.append(traj_actions)
                    trajs_reward.append(np.sum(traj_rewards, axis=0))
                    order_data = np.append(order_data, orders.T, axis=0)
                    allocation_data = np.append(allocation_data, allocations.T, axis=0)
                    reward_data = np.append(reward_data, np.array(traj_rewards).T, axis=0)
                    break

            # if not prestop:
            #     trajs_state.append(traj_states)
            #     trajs_action.append(traj_actions)
            #     trajs_reward.append(np.sum(traj_rewards))

        print(np.round(np.mean(trajs_reward)))

        traj_state_np = np.array(trajs_state)
        traj_action_np = np.array(trajs_action)

        # np.save('{}_{}_state'.format(config.env_name, config.task), traj_state_np)
        # np.save('{}_{}_action'.format(config.env_name, config.task), traj_action_np)
        print(traj_state_np.shape)
        print(traj_action_np.shape)
        np.savetxt(f'../render/ppo2/order_data_{config.condition}.csv', order_data, delimiter=',')
        np.savetxt(f'../render/ppo2/allocation_data_{config.condition}.csv',  allocation_data, delimiter=',')
        np.savetxt(f'../render/ppo2/rewards_data_{config.condition}.csv', reward_data, delimiter=',')


if __name__ == "__main__":
    main()
