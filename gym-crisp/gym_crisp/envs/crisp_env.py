import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from server import simulation_builder
from simulator.decision import TreatDecision
from simulator.decision import OrderDecision
from simulator.decision import ProduceDecision
from simulator.decision import AllocateDecision
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class CrispEnv2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=35, role='ds', study_name='study_2_1', start_cycle=60):
        self.n = n  # number of simulation periods in each episode
        self.np_random = None
        self.study_name = study_name
        self.start_cycle = start_cycle
        # self.data_columns = ['time', 'agent', 'item', 'value', 'unit']
        # self.data = pd.DataFrame(columns=self.data_columns)
        self.data = {
            'demand_hc1': [],
            'demand_hc2': [],
            'delivered-to-hc1': [],
            'delivered-to-hc2': []
        }
        self.simulation = None
        self.runner = None
        self.state = None
        self.agent = None
        self.decisions = []
        self.role = role  # the role of the agent
        self.backlog = 0
        self.reward = 0
        self.order = 0
        self.total_reward = 0
        self.allocation_n = 3
        self.min_order = 0
        self.max_order = 400
        self.observation_keys = ['inventory', 'demand-hc1', 'demand-hc2', 'on-order', 'shipment',
                                 'suggestion', 'outl', 'dlv-rate-hc1', 'dlv-rate-hc2', 'mn-inventory', 'disruption']
        # self.action_space = spaces.Box(
        #     low=self.min_order, high=self.max_order, shape=(1,), dtype=np.float32)
        # self.action_space = spaces.Discrete(1000)
        # self.action_space = spaces.Tuple([spaces.Discrete(3), spaces.Discrete(self.max_order)])
        self.action_space = spaces.MultiDiscrete([self.allocation_n, self.max_order])
        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0, 0, 0]), high=np.array([self.max_order, self.max_order, self.max_order, self.max_order]),
        #     dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #     name: (spaces.Box(shape=(1,), low=0, high=np.inf, dtype=np.float32)
        #            if name not in ['dlv-rate-hc1', 'dlv-rate-hc2', 'disruption']
        #            else spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32))
        #     for name in self.observation_keys
        # })
        self.observation_space = spaces.Dict(OrderedDict([
            (name, spaces.Box(shape=(1,), low=0, high=np.inf, dtype=np.float32)
                if name not in ['dlv-rate-hc1', 'dlv-rate-hc2', 'disruption']
                else spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32))
            for name in self.observation_keys
            ]))
        self.seed()
        # self.reset()

    def step(self, action):

        # reward = self.reward  # in this version reward only depends on allocation and inventory and not the #
        # agent's action
        self._take_action(action)
        self.runner._make_decision(self.simulation.now)
        self.agent = self._get_agent_by_role(self.role)
        self.agent.decisions = []
        self._parse_decisions()
        self.runner._apply_decision(self.simulation.now)

        # draw_figures(game, user_id, agent.name())

        self._collect_data()

        self.simulation.now += 1
        self.runner._update_patient(self.simulation.now)
        self.runner._update_network(self.simulation.now)
        self.runner._update_agents(self.simulation.now)
        self.runner._exogenous_event(self.simulation.now)

        self._reward()

        done = bool(self.simulation.now > 95)

        new_obs = self._get_obs(self.simulation.now)

        return new_obs, self.reward, done, {'time': self.simulation.now}

    def reset(self):
        self.total_reward = 0
        self.simulation = None
        self.runner = None
        self.simulation, self.runner = simulation_builder.build_simulation_study_2(self.study_name)
        self.runner._update_patient(0)
        self.runner._update_agents(0)
        self.agent = self._get_agent_by_role(self.role)
        self._fast_forward_simulation()

        self.simulation.now += 1
        self.runner._update_patient(self.simulation.now)
        self.runner._update_network(self.simulation.now)
        self.runner._update_agents(self.simulation.now)
        self.runner._exogenous_event(self.simulation.now)

        self.backlog = 0

        return self._get_obs(self.simulation.now)

    def render(self, mode='human'):
        # Render the environment to the screen

        self.total_reward += self.reward
        print(f'Simulation Time: {self.simulation.now - 1}')
        # print(f'Inventory: {self.agent.inventory_level()}')
        print(f'Inventory: {self.agent.history[self.simulation.now - 1]["inventory"]}')
        print(f'Backlog: {self.backlog}')
        # print(f'Order amount: {self.order}')
        print(
            f'Order amount: {[0 if not self.agent.history[self.simulation.now - 1]["order"] else self.agent.history[self.simulation.now - 1]["order"][0].amount]}')
        print(f'Reward: {self.reward}')
        print(f'total reward: {self.total_reward}')
        print('--------------------')

    def _reward(self):

        history_item = self.agent.get_history_item(self.simulation.now-1)

        backlog = int(sum(order.amount for order in self.agent.backlog)) - int(self.agent.demand(self.simulation.now))
        allocation = int(sum(alloc['item'].amount for alloc in history_item['allocate']))
        inventory = history_item.get('inventory') - allocation

        inventory_cost = int(inventory)
        backlog_cost = int(backlog) * 10
        sales = allocation * 5

        self.reward = sales-(inventory_cost + backlog_cost)
        return self.reward

    def _get_obs(self, now):

        inventory = self.agent.inventory_level()
        history_item = self.agent.get_history_item(now)
        demand = self.agent.demand(now)
        demand_hc1 = sum(in_order.amount for in_order in history_item['incoming_order']
                         if in_order.src.agent_name == 'HC1')
        demand_hc2 = sum(in_order.amount for in_order in history_item['incoming_order']
                         if in_order.src.agent_name == 'HC2')
        on_order = sum(order.amount for order in self.agent.on_order)
        shipment = sum(d['item'].amount for d in history_item['delivery'])
        backlog = sum(order.amount for order in self.agent.backlog)
        outl = self.agent.up_to_level
        mn_inventory = self.simulation.manufacturers[0].inventory_level()

        if backlog <= inventory:
            end_inventory = inventory - backlog
            end_backlog = 0
        else:
            end_inventory = 0
            end_backlog = backlog - inventory

        suggestion = max(outl + end_backlog - on_order - end_inventory, 0)

        self.backlog = end_backlog

        demand_hc1_c = sum(self.data.get('demand_hc1'))
        delivered_to_hc1_c = sum(self.data.get('delivered-to-hc1'))
        demand_hc2_c = sum(self.data.get('demand_hc2'))
        delivered_to_hc2_c = sum(self.data.get('delivered-to-hc2'))

        deliv_rate_to_hc1 = min(round(delivered_to_hc1_c / demand_hc1_c, 2), 1)
        deliv_rate_to_hc2 = min(round(delivered_to_hc2_c / demand_hc2_c, 2), 1)

        if self.simulation.disruptions and self.simulation.disruptions[0].happen_day_1 == self.simulation.now:
            disruption = 1
        else:
            disruption = 0

        return_obs = OrderedDict([
            ('inventory', [inventory]),
            ('demand-hc1', [demand_hc1]),
            ('demand-hc2', [demand_hc2]),
            ('on-order', [on_order]),
            ('shipment', [shipment]),
            ('suggestion', [suggestion]),
            ('outl', [outl]),
            ('dlv-rate-hc1', [deliv_rate_to_hc1]),
            ('dlv-rate-hc2', [deliv_rate_to_hc2]),
            ('mn-inventory', [mn_inventory]),
            ('disruption', [disruption])
        ])
        # return_obs = OrderedDict()
        # return_obs['inventory'] = [inventory]
        # return_obs['demand-hc1'] = [demand_hc1]
        # return_obs['demand-hc2'] = [demand_hc2]
        # return_obs['on-order'] = [on_order]
        # return_obs['shipment'] = [shipment]
        # return_obs['suggestion'] = [suggestion]
        # return_obs['outl'] = [outl]
        # return_obs['dlv-rate-hc1'] = [deliv_rate_to_hc1]
        # return_obs['dlv-rate-hc2'] = [deliv_rate_to_hc2]
        # return_obs['mn-inventory'] = [mn_inventory]
        # return_obs['disruption'] = [disruption]

        # return_obs = {
        #     'inventory': inventory,
        #     'demand-hc1': demand_hc1,
        #     'demand-hc2': demand_hc2,
        #     'on-order': on_order,
        #     'shipment': shipment,
        #     'suggestion': suggestion,
        #     'outl': outl,
        #     'dlv-rate-hc1': deliv_rate_to_hc1,
        #     'dlv-rate-hc2': deliv_rate_to_hc2,
        #     'mn-inventory': mn_inventory,
        #     'disruption': disruption
        # }

        # return np.array([inventory, shipment, demand, backlog])
        return return_obs

    def _get_agent_by_role(self, role):
        #   IMPORTANT:  This part probably needs to be edited. Right now this function only returns the first agent in
        #               the list from simulation but in a more complex network it should check to see which
        #               agent is gonna be the RL agent and return that one.

        agent_list = None

        if role == 'hc':
            agent_list = self.simulation.health_centers
        elif role == 'ws':
            agent_list = [k for k in self.simulation.distributors if k.agent_name == 'WS']
        elif role == 'ds':
            agent_list = [k for k in self.simulation.distributors if k.agent_name == 'DS1']
        elif role == 'mn':
            agent_list = self.simulation.manufacturers

        return agent_list[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _rescale_action(self, action):
        rescaled_actions = np.zeros((2,), dtype=np.int64)

        bin_1 = np.arange(-1, 1, 2/self.allocation_n, dtype=np.float32)
        bin_2 = np.arange(-1, 1, 2/self.max_order, dtype=np.float32)

        rescaled_actions[0] = np.digitize(action[0], bin_1[1:])
        rescaled_actions[1] = np.digitize(action[1], bin_2[1:])

        return rescaled_actions

    def _take_action(self, action):

        if isinstance(action[0], np.float32) or isinstance(action[1], np.float32):
            action = self._rescale_action(action)

        inventory = self.agent.inventory_level()
        backlog = sum(order.amount for order in self.agent.backlog)

        backlog_hc1 = sum(order.amount for order in self.agent.backlog
                          if order.src == self.simulation.health_centers[0])
        backlog_hc2 = sum(order.amount for order in self.agent.backlog
                          if order.src == self.simulation.health_centers[1])

        alloc_hc1, alloc_hc2 = 0, 0

        if backlog <= inventory:
            alloc_hc1 = backlog_hc1
            alloc_hc2 = backlog_hc2
        else:
            if int(action[0]) == 0:
                if inventory >= backlog_hc1:
                    alloc_hc1 = backlog_hc1
                    inventory -= backlog_hc1
                    alloc_hc2 = inventory
                else:
                    alloc_hc1 = inventory
                    alloc_hc2 = 0
            elif int(action[0]) == 1:
                if inventory >= backlog_hc2:
                    alloc_hc2 = backlog_hc2
                    inventory -= backlog_hc2
                    alloc_hc1 = inventory
                else:
                    alloc_hc2 = inventory
                    alloc_hc1 = 0
            elif int(action[0]) == 2:
                alloc_hc1 = int(round(inventory * backlog_hc1 / backlog, 0))
                # alloc_hc2 = int(round(inventory * backlog_hc2 / backlog, 0))
                alloc_hc2 = inventory - alloc_hc1

        self.decisions = [
            {
                'agent': self.agent,
                'decision_name': 'satisfy_hc1',
                'decision_value': int(alloc_hc1),
            },
            {
                'agent': self.agent,
                'decision_name': 'satisfy_hc2',
                'decision_value': int(alloc_hc2),
            },
            {
                'agent': self.agent,
                'decision_name': 'order_from_mn1',
                'decision_value': int(action[1]),
                # 'decision_value': int(120),
            }
        ]

        # self.decisions.append(decision)
        self.order = action[1]

    def _parse_decisions(self):
        """ convert the game decision to simulator desicion """
        for decision in self.decisions:
            self._convert_to_simulation_decision(decision)

        self.decisions = []

    def _convert_to_simulation_decision(self, agent_decision):
        """
        :type agent_decision: RL Agent Decision
        """

        if agent_decision['decision_name'] == 'satisfy_urgent':
            decision = TreatDecision()
            decision.urgent = agent_decision['decision_value']
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_non_urgent':
            decision = TreatDecision()
            decision.non_urgent = agent_decision['decision_value']
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_ds1':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            # decision.upstream = self.simulation.distributors[0]
            decision.upstream = [k for k in self.simulation.distributors if k.agent_name == 'DS'][0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_ds2':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            # decision.upstream = self.simulation.distributors[1]
            decision.upstream = [k for k in self.simulation.distributors if k.agent_name == 'DS'][1]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_mn1':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            decision.upstream = self.simulation.manufacturers[0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'order_from_mn2':
            decision = OrderDecision()
            decision.amount = agent_decision['decision_value']
            decision.upstream = self.simulation.manufacturers[1]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'produce':
            decision = ProduceDecision()
            decision.amount = agent_decision['decision_value']
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_ds1':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.distributors[0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_ds2':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.distributors[1]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_hc1':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.health_centers[0]
            agent_decision['agent'].decisions.append(decision)

        elif agent_decision['decision_name'] == 'satisfy_hc2':
            decision = AllocateDecision()
            decision.amount = agent_decision['decision_value']
            decision.downstream_node = self.simulation.health_centers[1]
            agent_decision['agent'].decisions.append(decision)

        else:
            print("Decision type " + agent_decision['decision_name']
                  + " not supported!\n")
            return

    def _fast_forward_simulation(self):
        """ Let the default decision maker to run the game for a certain number of
            cycles

            :param cycle: the starting cycle of the simulation
        """
        # reward = 0
        for i in range(0, self.start_cycle):
            self.runner.next_cycle()
            # if i > 61:
            #     self.agent = self._get_agent_by_role(self.role)
            #     reward += self._reward()
            if i >= self.start_cycle - 9:
                self._collect_data()

    def _collect_data(self):

        now = self.simulation.now
        history = self.agent.get_history_item(now)

        for key, my_array in self.data.items():
            if key == 'demand_hc1':
                self.data.get(key).append(sum(in_order.amount for in_order in history['incoming_order']
                                              if in_order.src.agent_name == 'HC1'))
            elif key == 'demand_hc2':
                self.data.get(key).append(sum(in_order.amount for in_order in history['incoming_order']
                                              if in_order.src.agent_name == 'HC2'))
            elif key == 'delivered-to-hc1':
                self.data.get(key).append(sum([allocation['item'].amount for allocation in history['allocate']
                                               if allocation['order'].src.agent_name == 'HC1']))
            elif key == 'delivered-to-hc2':
                self.data.get(key).append(sum([allocation['item'].amount for allocation in history['allocate']
                                               if allocation['order'].src.agent_name == 'HC2']))
            else:
                raise KeyError(f'No stat collection is defined for key {key}')

            if len(my_array) > 9:
                del self.data.get(key)[0]
