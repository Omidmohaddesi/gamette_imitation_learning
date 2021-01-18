import numpy as np
import sys
import os


class Buffer:
    def __init__(self, config, expert_traj, expert_theta, policy, env):
        self.config = config
        self.policy = policy
        self.env = env

        self.expert_traj = expert_traj
        self.expert_theta = expert_theta
        self.num_expert_traj = len(self.expert_traj)
        print('num expert traj: ', self.num_expert_traj)

        # self.empty_code = np.array([-1] * config.batch_size_traj)

    def sample_expert_traj(self):
        # expert_traj_idx = np.random.choice(range(2000, 4000), size=self.config.batch_size_traj, replace=False)
        expert_traj_idx = np.arange(self.expert_traj.shape[0])
        np.random.shuffle(expert_traj_idx)
        expert_traj_state_sampled = self.expert_traj[expert_traj_idx]
        expert_traj_action_sampled = self.expert_theta[expert_traj_idx]

        return expert_traj_state_sampled, expert_traj_action_sampled

    # def sample_code(self):
    #     return np.random.randint(self.config.num_code, size = self.config.batch_size_traj) \
    #         if not self.config.empty_code else self.empty_code

    def sample_stu_traj(self):
        stu_traj_states = []
        stu_traj_actions = []
        stu_traj_code_np = np.random.randint(self.config.num_code, size = self.config.batch_size_traj)

        init_h_state = None
        init_state = self.env.reset()
        curr_h_state = init_h_state
        # curr_state = init_state[:, :self.config.state_dim]
        curr_state = init_state
        # make sure the order of keys are the same:
        # curr_state = type(init_state)((k, init_state[k]) for k in self.config.obs_filter_keys)

        rewards = []

        # rollout trajectory
        for _ in range(self.config.max_traj_len):
            action_sampled, curr_h_state = self.policy.sample_action(curr_state, stu_traj_code_np, curr_h_state)
            stu_traj_states.append(curr_state)
            stu_traj_actions.append(action_sampled)

            next_state, reward, done, info = self.env.step(action_sampled)
            # curr_state = next_state[:, :self.config.state_dim]
            curr_state = next_state
            # make sure the order of keys are the same:
            # curr_state = type(next_state)((k, next_state[k]) for k in self.config.obs_filter_keys)

            rewards.append(reward)

        # stu_traj_state_np = np.stack([np.array(list(d.values())).T for d in stu_traj_states], axis=1)
        stu_traj_state_np = np.stack(stu_traj_states, axis=1)
        stu_traj_action_np = np.stack(stu_traj_actions, axis=1)

        reward_np = np.sum(np.array(rewards), axis=0)

        return stu_traj_state_np, stu_traj_action_np, stu_traj_code_np, reward_np
