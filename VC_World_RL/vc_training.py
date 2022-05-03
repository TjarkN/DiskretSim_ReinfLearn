from vc_environment import Environment
from agents.q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np

# initialize Environment and agent
def initialize_env(x,y):
    env = Environment(size=(x, y), seed_value=None)
    agent = QLearningAgent(env.problem)
    states = env.problem.get_all_states()
    actions = env.problem.get_all_actions()

    #inputs
    percept = [env.problem.get_current_state(), env.problem.get_reward()]

    #persistent
    q = []
    n = []
    s = None
    a = None

    if s != None:
        for i in n:
            q = q + np.a
    #while true
    #    env.reset()

    q_table = np.zeros(len(states),len(actions))

# training

# test_agent

# plot results

# save q_table
