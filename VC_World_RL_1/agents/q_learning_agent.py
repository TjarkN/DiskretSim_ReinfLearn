from agents.agent import Agent
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


class QLearningAgent(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100,
                 q_table_file="q_table.npy"):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = {}
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = {}
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max
        self.file = q_table_file
        #self.experience_replay = ExperienceReplay()

    def act(self):
        # perception
        s = self.problem.get_current_state().to_state()
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):
        action = None
        s_new = None
        while True:
            current_state = self.problem.get_current_state()
            r = self.problem.get_reward(current_state)
            s = s_new
            s_new = current_state.to_state()
            if s_new not in self.N_sa.keys():
                self.N_sa[s_new] = np.zeros(len(self.actions))
                self.q_table[s_new] = np.zeros(len(self.actions))
            if action is not None:
                a = self.actions.index(action)
                self.N_sa[s][a] += 1
                self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(current_state))
            if self.problem.is_goal_state(current_state):
                return self.q_table, self.N_sa
            action = self.choose_GLIE_action(self.q_table[s_new], self.N_sa[s_new])
            # act
            self.problem.act(action)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        if is_goal_state:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha(s, a) * (r-self.q_table[s][a])
        else:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha(s, a) * (r + self.gamma * np.max(self.q_table[s_new]) -
                                                                      self.q_table[s][a])

    def choose_GLIE_action(self, q_values, N_s):
        exploration_values = np.ones_like(q_values) * self.R_Max
        # which state/action pairs have been visited sufficiently
        no_sufficient_exploration = N_s < self.max_N_exploration
        # turn cost to a positive number
        q_values_pos = self.R_Max / 2 + q_values
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)
        # assure that we do not dived by zero
        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values) / max_values.size
        else:
            probabilities = max_values / max_values.sum()
        # select action according to the (q) values
        if np.random.random() < (self.max_N_exploration+0.00001)/(np.max(N_s)+0.00001):
            action = np.random.choice(self.actions, p=probabilities)
        else:
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes)
            action = self.actions[action_index]
        return action

    def save_q_table(self):
        np.save(self.file, self.q_table)

    def load_q_table(self):
        self.q_table = np.load(self.file) #, allow_pickle=True

    def alpha(self, s, a):
        # learnrate alpha decreases with N_sa for convergence
        alpha = self.N_sa[s][a]**(-1/2)
        return alpha



class ExperienceReplay(Dataset):
    def __init__(self, model, max_memory=100, gamma=0.99, transform=None, target_transform=None):
        self.model = model
        self.memory = []
        self.max_memory = max_memory
        self.gamma = gamma
        self.transform = transform
        self.target_transform = target_transform

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def update_model(self, model):
        self.model = model

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        s, a, r, s_new = self.memory[idx][0]
        goal_state = self.memory[idx][1]
        features = np.array(s)
        # init labels with old prediction (and later overwrite action a)
        label = self.model[s]
        if goal_state:
            label[a] = r
        else:
            label[a] = r + self.gamma * max(self.model[s_new])

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        features = torch.from_numpy(features).float().to(device)
        label = torch.from_numpy(label).float().to(device)

        return features, label


class DeepQLearningAgent(QLearningAgent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100, batch_size=10,
                 Optimizer=torch.optim.Adam, loss_fn=nn.MSELoss()):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration,
                         R_Max=R_Max)

        if q_table is None:
            all_states = np.array(self.states)
            min_values = np.amin(all_states, axis=0)
            max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
            transform = lambda x: (x - min_values) / (max_values - min_values)
            self.q_table = self.create_model(Optimizer, loss_fn, transform)

        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table, transform=transform)
        self.loss_history = []


    #def __call__(self, *args, **kwargs):

    def create_model(self, Optimizer, loss_fn, transform):
        return DeepQTable(len(self.states[0]), len(self.actions), Optimizer, loss_fn, transform)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)
        train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
        self.loss_history += self.q_table.perform_training(train_loader)



class DeepQTable(nn.Module):
    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepQTable, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, number_of_actions), )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)
        self.optimizer = Optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        self.transform = transform

    def __getitem__(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().to(self.device)
        return self.model(state).cpu().detach().numpy()

    def __setitem__(self, state, value):
        # ignoring setting to values
        pass

    def forward(self, x):
        return self.model(x)

    def perform_training(self, dataloader):
        loss_history = []
        (X, y) = next(iter(dataloader))
        # Compute prediction and loss
        pred = self(X)
        loss = self.loss_fn(pred, y)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_history.append(loss)
        return loss_history

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))



"""class ExperienceReplay(Dataset):

    def __init__(self):
        self.data = []

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def remember(self, e):
        self.data.append(e)"""


