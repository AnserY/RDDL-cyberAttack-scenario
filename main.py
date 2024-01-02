import random
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.Agents import RandomAgent, BaseAgent
from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer

from pyRDDLGym.Visualizer.TextViz import TextVisualizer
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
from pprint import pprint
import numpy as np



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



    

env =  RDDLEnv.RDDLEnv(domain='domain.rddl', instance='local_instance.rddl')

print(env.set_visualizer(TextVisualizer))

#print(len(env.action_space))

#agent = QLearningAgent(action_space=env.action_space,state_space_size=5 ,learning_rate=0.05, discount_factor=0.99, exploration_rate=0.1)

agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)


#pprint(vars(env.model))

total_reward = 0
state = env.reset()
for step in range(env.horizon):
    env.render()
    action = agent.sample_action(state)
    next_state, reward, done, info = env.step(action)
    print("\done = {}".format(done))
    print("\nstep = {}".format(step))
    print('reward = {}'.format(reward))
    print('state = {}'.format(state))
    print('action = {}'.format(action))
    print('info = {}'.format(info))
    print('next_state = {}'.format(next_state))
    #print(len(env.action_space))


    
    #print('derived = {}'.format(env.model.derived))
    #print('interm = {}'.format(env.model.interm))

    
    #print(f'state = {state}, action = {action}, reward = {reward}')
    total_reward += reward
    state = next_state
    if done:
        break
print(f'episode ended with reward {total_reward}')

# release all viz resources, and finish logging if used
env.close()
