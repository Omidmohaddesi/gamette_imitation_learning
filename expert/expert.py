import gym
import numpy as np
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym.wrappers.filter_observation import FilterObservation
from stable_baselines import PPO2
import os
from tqdm import tqdm
import pdb
import sys
from config import config

import gym_crisp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


crispPath = os.path.join(os.path.abspath('../..'), 'crisp/')
crisp_server_Path = os.path.join(os.path.abspath('../..'), 'crisp_game_server/')
sys.path.insert(1, crispPath)
sys.path.insert(1, crisp_server_Path)


def scale_orders(my_df):
    """
    transforms order_data dataframe of experts (i.e. players) to the range [-1, 1] by dividing each element
    by maximum order across all players
    :param my_df: order_data dataframe
    :return: transformed dataframe in range [-1, 1]
    """
    return (((my_df * 2) / my_df.max().max()) - 1).astype(np.float32)


def convert_actions(my_df):

    new_df = my_df.copy()
    new_df.loc[:, range(21, 56)] = pd.DataFrame(
        np.where(new_df[range(21, 56)] == 'Prefer HC1', -1.,
                 np.where(new_df[range(21, 56)] == 'Prefer HC2', 0., 1.)),
        columns=range(21, 56))

    return new_df.astype(np.float32)


def main():

    order_data = pd.read_csv(os.path.join(config.expert_data_path, 'order_data.csv'), index_col=0)
    allocation_data = pd.read_csv(os.path.join(config.expert_data_path, 'allocation_data.csv'), index_col=0)

    players = pd.read_csv(config.players_to_consider_path, index_col=0)

    order_data = order_data[(order_data['player_id'].isin(players.values.ravel())) &
                            (order_data['condition'] == config.condition)].reset_index(drop=True).drop(
        ['player_id', 'condition'], axis=1)
    allocation_data = allocation_data[(allocation_data['player_id'].isin(players.values.ravel())) &
                                      (allocation_data['condition'] == config.condition)].reset_index(drop=True).drop(
        ['player_id', 'condition'], axis=1)

    order_data.columns = order_data.columns.astype(int)
    allocation_data.columns = allocation_data.columns.astype(int)

    allocation_data = convert_actions(allocation_data)

    scaled_order_data = scale_orders(order_data)

    if config.n_cpu == 1:
        env = FilterObservation(gym.make(config.env_name,
                                         study_name=config.study_name,
                                         start_cycle=config.start_cycle),
                                filter_keys=config.obs_filter_keys)
        # env = DummyVecEnv([lambda: FilterObservation(gym.make(config.env_name,
        #                                                       study_name=config.study_name,
        #                                                       start_cycle=config.start_cycle),
        #                                              filter_keys=config.obs_filter_keys)])
    else:
        env = SubprocVecEnv([lambda: FilterObservation(gym.make(config.env_name,
                                                                study_name=config.study_name,
                                                                start_cycle=config.start_cycle),
                                                       filter_keys=config.obs_filter_keys)
                             for _ in range(config.n_cpu)])

    trajs_state = []
    trajs_action = []
    trajs_reward = []

    traj_state_np = np.array([])
    traj_action_np = np.array([])

    for orders, allocations in tqdm(zip(scaled_order_data.iterrows(), allocation_data.iterrows())):

        # prestop = False
        traj_states = []
        traj_actions = []
        traj_rewards = []

        obs = env.reset()

        for i in range(config.episode_start, config.episode_end + 1):
            action = np.array([allocations[1][i], orders[1][i]])
            # action = env.action_space.sample()

            traj_states.append(obs)
            traj_actions.append(action)

            obs, rewards, done, info = env.step(action)
            # print(obs, rewards, dones, info)
            # env.render()
            traj_rewards.append(rewards)
            # traj_rewards.append(info[0]['backward_reward'])

            if done:
                # prestop = True
                trajs_state.append(traj_states)
                trajs_action.append(traj_actions)
                trajs_reward.append(np.sum(traj_rewards))
                break

        # if not prestop:
        #     trajs_state.append(traj_states)
        #     trajs_action.append(traj_actions)
        #     trajs_reward.append(np.sum(traj_rewards))

        print(np.mean(trajs_reward))

        traj_state_np = np.array(trajs_state)
        traj_action_np = np.array(trajs_action)

    np.save('traj/{}_{}_state'.format(config.env_name, config.condition), traj_state_np)
    np.save('traj/{}_{}_action'.format(config.env_name, config.condition), traj_action_np)
    print(traj_state_np.shape)
    print(traj_action_np.shape)
    print()
    print('Max order amount among all players: ', order_data.max().max())


if __name__ == "__main__":
    main()
