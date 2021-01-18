from easydict import EasyDict as edict
import os


config = edict()

config.task = 'stand'
config.env_name = 'Crisp-v2'

config.condition = 3
config.study_name = 'study_2_3'
config.start_cycle = 60
config.obs_filter_keys = ['inventory', 'demand-hc1', 'demand-hc2', 'on-order', 'shipment', 'suggestion', 'outl']
config.episodes = 35
config.episode_start = 21
config.episode_end = 55
config.n_step = 36


config.expert_data_path = os.path.join(os.path.abspath('../..'),
                                       'crisp_game_server/gamette_experiments/study_2/data_full')
config.players_to_consider_path = os.path.join(os.path.abspath(''), 'players.csv')

# if config.task == 'forward':
#     config.forward_reward_weight = 1.0
#     config.ctrl_cost_weight = 1e-3
# elif config.task == 'stand':
#     config.forward_reward_weight = 0.0
#     config.ctrl_cost_weight = 1.0
# elif config.task == 'backward':
#     config.forward_reward_weight = -1.0
#     config.ctrl_cost_weight = 1e-3
# else:
#     raise NotImplementedError

# config.terminate_when_unhealthy = False
config.test = True
config.load = True
config.ckpt_path = config.task + '_ckpt'

config.itr = 5000000
config.sample_size = 2000

config.verbose = 1
config.n_cpu = 1
config.n_minibatch = 8
config.n_optepoch = 4
