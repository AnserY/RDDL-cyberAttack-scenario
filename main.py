import random
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.Agents import RandomAgent, BaseAgent
from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer

from pyRDDLGym.Visualizer.TextViz import TextVisualizer
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
from pprint import pprint
import numpy as np
import pandas as pd
from itertools import islice


previous_values = {'hack___h1': True, 'hack___h2': False, 'hack___h3': False, 'hack___h4': False, 'hack___h5': False, 'hack___h6': False, 'hack___h7': False, 'hack___h8': False, 'hack___h9': False, 'hack___h10': False, 'phish___p1': False, 'phish___p2': False, 'phish___p3': False, 'phish___p4': False, 'phish___p5': False, 'phish___p6': False, 'phish___p7': False, 'phish___p8': False, 'phish___p9': False, 'phish___p10': False}
d = {'node_1': 'h1', 'node_2': 'h2', 'node_3': 'h3', 'node_4': 'h4', 'node_5': 'h5', 'node_6': 'h6', 'node_7': 'h7', 'node_8': 'h8', 'node_9': 'h9', 'node_10': 'h10', 'node_11': 'p1', 'node_12': 'p2', 'node_13': 'p3', 'node_14': 'p4', 'node_15': 'p5', 'node_16': 'p6', 'node_17': 'p7', 'node_18': 'p8', 'node_19': 'p9', 'node_20': 'p10', 'node_1_type':'host', 'node_2_type':'host','node_3_type':'host', 'node_4_type':'host','node_5_type':'host',
        'node_6_type':'host','node_7_type':'host','node_8_type':'host','node_9_type':'host', 'node_10_type':'host', 'node_11_type':'password', 'node_12_type':'password' , 'node_13_type':'password' , 'node_14_type':'password' , 'node_15_type':'password' , 'node_16_type':'password' , 'node_17_type':'password' , 'node_18_type':'password' , 'node_19_type':'password' ,
        'node_20_type':'password','edge_1_type':'CONNECTED','edge_1_prev':'h1','edge_1_suiv':'h5','edge_2_type':'CONNECTED','edge_2_prev':'h2','edge_2_suiv':'h4','edge_3_type':'CONNECTED','edge_3_prev':'h3','edge_3_suiv':'h10','edge_4_type':'CONNECTED','edge_4_prev':'h4','edge_4_suiv':'h7','edge_5_type':'CONNECTED', 'edge_5_prev':'h5','edge_5_suiv':'h2' ,'edge_6_type':'CONNECTED','edge_6_prev': 'h6', 'edge_6_suiv':'h4', 'edge_7_type': 'CONNECTED', 'edge_7_prev': 'h7', 'edge_7_suiv': 'h10','edge_8_type':'CONNECTED','edge_8_prev':'h8','edge_8_suiv':'h4','edge_9_type':'ACCESS', 'edge_8_prev':'p1','edge_8_suiv':'h1' ,'node_1_label' : 'hacked', 'node_2_label': 'not hacked' , 'node_3_label': 'not hacked' ,
        'node_4_label': 'not hacked' , 'node_5_label': 'not hacked' , 'node_6_label': 'not hacked' , 'node_7_label': 'not hacked' , 'node_8_label': 'not hacked' , 'node_9_label': 'not hacked', 'node_10_label': 'not hacked', 'node_11_label':'not phished' , 'node_12_label':'not phished' , 'node_13_label':'not phished' , 'node_14_label':'not phished' , 'node_15_label':'not phished' , 'node_16_label':'not phished' , 'node_17_label':'not phished' , 'node_18_label':'not phished' ,
        'node_19_label':'not phished' , 'node_20_label':'not phished' , 'Action_attempted':None }



def first_n_items(input_dict, n=20):
    return dict(islice(input_dict.items(), n))


def detect_changes(current_values, previous_values):
    changed_keys = []
    for key, value in current_values.items():
        # Check if the current value is True and the previous value was False
        if value and not previous_values.get(key, False):
            changed_keys.append(key)
    return changed_keys


class QLearningAgent:
    def __init__(self, action_space,state_space_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.action_space = action_space
        self.state_space_size  = state_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = np.zeros((state_space_size, len(action_space)))

    def sample_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            return self.action_space.sample()  # Explore

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value





env =  RDDLEnv.RDDLEnv(domain='domain-v2.rddl', instance='local_instance.rddl')
env.set_visualizer(TextVisualizer)

df = pd.DataFrame([d])

#agent = QLearningAgent(action_space=env.action_space,state_space_size=5 ,learning_rate=0.05, discount_factor=0.99, exploration_rate=0.1)

agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)

total_reward = 0
state = env.reset()
for step in range(env.horizon):
    env.render()
    action = agent.sample_action(state)
    next_state, reward, done, info = env.step(action)
    if (env.domain_text=="  domain-v1") : 
        print("\done = {}".format(done))
        print("\nstep = {}".format(step))
        print('reward = {}'.format(reward))
        print('state = {}'.format(state))
        print('action = {}'.format(action))
        print('info = {}'.format(info))
        print('next_state = {}'.format(next_state))
    else :
        current_values = first_n_items(state)

        change = detect_changes(current_values, previous_values)
        for item in change :
            if item.startswith('hack'):
                suffix = item.split('__h')[1]
                d[f'node_{suffix}_label']= 'hacked'
            else:
                suffix = item.split('__p')[1]
                d[f'node_{int(suffix) + 10}_label'] = 'phished'

        d['Action_attempted'] = next(iter(action))
        df = df.append(d, ignore_index=True)
        previous_values = current_values
    
    total_reward += reward
    state = next_state
    if done:
        break
print(f'episode ended with reward {total_reward}')

df.to_csv("gnn.csv", index=False)
env.close()
