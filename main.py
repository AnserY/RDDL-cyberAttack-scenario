import random
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.Agents import RandomAgent, BaseAgent
from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer





class DirectAgent(BaseAgent):
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)

    def action_horizon(self, state):
        hackable = []
        for key, value in state.items():
            if key.startswith("hackable") and value:
                action_key = key.replace("hackable", "")
                hackable.append(action_key)               
        return hackable

    def sample_action(self, state=None):
        possible_actions = self.action_horizon(state)
        s = self.action_space.sample()
        action = {}
        if len(possible_actions) > 0:
            selected_action = random.choice(possible_actions)
            action[selected_action] = 1
        else:
            action[list(s.keys())[0]] = 0
        return action
    

env = RDDLEnv.RDDLEnv(domain='domain.rddl', instance='local_instance.rddl')
env.set_visualizer(ChartVisualizer)

agent = RandomAgent(action_space=env.action_space, seed=3)

total_reward = 0
state = env.reset()
for step in range(env.horizon):
    env.render()
    action = agent.sample_action(state)
    next_state, reward, done, info = env.step(action)
    print(f'state = {state}, action = {action}, reward = {reward}')
    total_reward += reward
    state = next_state
    if done:
        break
print(f'episode ended with reward {total_reward}')

# release all viz resources, and finish logging if used
env.close()
