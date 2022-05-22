from vc_environment import Environment
from agents.q_learning_agent import QLearningAgent
from agents.q_learning_agent import DeepQLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn

t = time.time()
# set max number of iterations
max_iterations = 2000
size = (3, 2)
it = 0
env = Environment(size)
agent = DeepQLearningAgent(env.problem, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100, batch_size=10,
                           Optimizer=torch.optim.Adam, loss_fn=nn.MSELoss())
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
run_time = time.time()-t
print(run_time)
print(env.problem.energy_spend)
x = np.array(performance)
N = 100
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance)
plt.plot(moving_average)
plt.show()

# save q_table
agent.q_table.save_model("2022_05_21.pth")
#agent.save_q_table()
