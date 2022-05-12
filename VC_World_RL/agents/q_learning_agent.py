import random

from .agent import Agent
import numpy as np


class QLearningAgent(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = np.zeros((len(self.states), (len(self.actions))))
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = np.zeros((len(self.states), (len(self.actions))))
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max

    def act(self):
        # perception
        current_state = self.problem.get_current_state()
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):
        s_act = self.problem.get_current_state()
        act_index = self.states.index(s_act.to_state())
        #r = self.problem.get_reward(s1)
        alpha = 0.3
        epsilon = 0.3

        #print (s1, r)
        #q_table = ()
        N_sa = ()

        while not self.problem.is_goal_state(self.problem):
            if random.random() < epsilon:
                appl_actions = self.problem.get_applicable_actions(s_act)
                action = np.random.choice(appl_actions)
            else:
                action = self.actions[np.argmax(self.q_table[act_index])]

            old_reward = self.problem.get_reward(s_act)
            q = self.q_table[act_index][self.actions.index(action)]
            self.problem.act(action)

            s_new = self.problem.get_current_state()
            new_index = self.states.index(s_new.to_state())

            r = self.problem.get_reward(s_new) - old_reward

            a_new = self.actions[np.argmax(self.q_table[act_index])]
            new_Q_max = max(self.q_table[new_index])

            q_value_new = q + alpha*(r + self.gamma * new_Q_max - q)
            self.q_table[act_index][self.actions.index(action)] = q_value_new

            s_act = s_new
            act_index = self.states.index(s_act.to_state())

        return self.q_table, N_sa

    def save_q_table(self, file):
        np.save(file, self.q_table)

    def load_q_table(self, file):
        self.q_table = np.load(file)
