import torch
import torch.nn as nn

from .deep_q_learning_agent import DeepQLearningAgent, DeepQTable
from .reinforce_agent import ReinforceAgent, PolicyNetwork


# Funktioniert leider noch nicht

class QActorPolicyNetwork(PolicyNetwork):

    def __init__(self, num_inputs, num_actions, Optimizer=torch.optim.Adam, learning_rate=3e-4,
                 gamma=0.99, transform=None):

        super(QActorPolicyNetwork, self).__init__(num_inputs, num_actions, Optimizer, learning_rate,
                                                  gamma, transform)

    def update_policy(self, log_probabilities, rewards, states):
        policy_gradient = []
        for log_prob, s in zip(log_probabilities, states):
            policy_gradient.append(-log_prob * self.q_table.__getitem__(s))

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

class QActorCriticAgent(ReinforceAgent, DeepQLearningAgent):

    def __init__(self, problem, Optimizer=torch.optim.Adam, gamma=0.99, file="policy.npy", PolicyClass= QActorPolicyNetwork,
                 q_table = None, N_sa = None, max_N_exploration = 100, R_Max = 100,
                 q_table_file = "deep_q_table.pth", batch_size = 10, loss_fn = nn.MSELoss(), ModelClass = DeepQTable):
        ReinforceAgent.__init__(self, problem =problem, Optimizer=Optimizer, gamma=gamma, file =file,PolicyClass= PolicyClass)
        DeepQLearningAgent.__init__(self, problem, q_table, N_sa, gamma, max_N_exploration, R_Max, q_table_file, batch_size,
                                    Optimizer, loss_fn, ModelClass)


    def train(self):
        log_probabilities = []
        rewards = []
        states = []
        #action = None
        s_new = None
        while True:
            current_state = self.problem.get_current_state()
            states.append(current_state)
            #s = s_new
            rewards.append(self.problem.get_reward(current_state))
            s = current_state.to_state()
            action_index, log_prob = self.policy.get_action(s)
            log_probabilities.append(log_prob)
            if self.problem.is_goal_state(current_state):
                if len(rewards) > 1:
                    self.policy.update_policy(log_probabilities, rewards, states)
                return
            # act
            action = self.actions[action_index]
            self.problem.act(action)

            r = self.problem.get_reward(current_state)
            s_new = self.problem.get_current_state().to_state()
            a = self.actions.index(action)
            self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(current_state))

