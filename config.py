from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os

dir_name = os.path.dirname(__file__)

config = edict()

config.env_name = 'Crisp-v2'

config.condition = 3
config.study_name = 'study_2_3'
config.start_cycle = 60
config.obs_filter_keys = ['inventory', 'demand-hc1', 'demand-hc2', 'on-order', 'shipment', 'suggestion', 'outl']
config.episodes = 35
config.episode_start = 21
config.episode_end = 55
config.n_step = 36

config.greedy = False
config.gpu = True
config.sess_nan_test = False
config.mode = 'train'
# config.mode = 'test'

config.activation = tf.nn.elu
config.normalize_adv = True
config.scale_action = False

config.itr = 500      # 500
config.test_itr = 32    # 32
# config.max_traj_len = 128
config.max_traj_len = 35
config.update_period = 5
config.inner_itr_1 = 2
config.inner_itr_2 = 5
config.print_itr = 5    # 5
config.save_itr = 10     # 10
config.batch_size_traj = config.n_cpu = 8
if config.mode == 'render':
    config.batch_size_traj = config.n_cpu = 1

config.save_path = os.path.join(dir_name, './ckpt/contion_{}.ckpt'.format(config.condition))
config.load_path = os.path.join(dir_name, './ckpt/contion_{}.ckpt'.format(config.condition))

config.expert_traj_prefix = os.path.join(dir_name, 'expert', 'Hopper-v3')

# config.state_dim = 5
config.state_dim = len(config.obs_filter_keys)
config.action_dim = 2
config.code_dim = config.num_code = 3
# config.action_range = 1.0
config.action_range = [3, 400]
config.action_high = 1.0
config.action_low = -1.0

config.gamma = 0.99
config.lam = 0.95

config.hidden_dim = 128
config.state_code_arch = 'add'
config.state_fc_dims = [128, 128]
config.code_fc_dims = [128]
config.action_fc_dims = [32]

config.policy_lr = 2e-4
config.policy_clip_range = 0.2
config.policy_log_std_init = -1.0
config.entropy_loss_coef = 0.0
config.policy_fc_dims = [128]

config.policy_log_std_mode = 'variable'
config.start_log_std = -2.0
config.end_log_std = -2.0
config.anneal_step = 10000

config.value_lr = 5e-4
config.value_clip_range = 1.0
config.value_fc_dims = [128, 128] # [128]

config.dis_lr = 2e-4
config.dis_coef = 1.0
config.dis_weight_clip = 0.01
config.dis_fc_dims = [128]

config.post_lr = 2e-4
config.post_coef = 1.0     # >>>>>>>> it was 0
config.post_fc_dims = [128]
