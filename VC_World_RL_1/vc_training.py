from vc_environment import Environment
from agents.q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np

# set max number of iterations
max_iterations = 2000
size = (3, 2)
it = 0
env = Environment(size)
agent = QLearningAgent(env.problem, max_N_exploration=10, file="agents/q_table.npy")
performance = []
q_table = None
# training
while it < max_iterations:
    complexity = max(1, env.problem.eval(env.problem))
    q_table, N_sa = agent.train()
    energy_spent = env.problem.energy_spend
    performance.append(energy_spent/complexity)
    print(it, performance[it])
    it += 1
    env.reset()

# test_agent
#env = Environment(size)
#agent = QLearningAgent(env.problem, q_table)
#while not env.problem.is_goal_state(env.problem):
#    action = agent.act()
#    env.problem.act(action)

# plot results
print(env.problem.energy_spend)
x = np.array(performance)
N = 100
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance)
plt.plot(moving_average)
plt.show()

# save q_table
agent.save_q_table()
