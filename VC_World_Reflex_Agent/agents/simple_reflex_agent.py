from .agent import Agent
import numpy as np


class SimpleReflexAgent(Agent):

    def __init__(self, problem, q_table=None):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = self.create_q_table()

    def act(self):
        # perception
        current_state = self.problem.get_current_state()
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def create_q_table(self):
        q_table = np.zeros((len(self.states), (len(self.actions))))

        #Hard coded q_table for a 1x2 board
        """q_table[0, 0] = 1
        q_table[1, 0] = 1
        q_table[2, 0] = 1
        q_table[3, 2] = 1
        q_table[4, 3] = 1
        q_table[5, 0] = 1
        q_table[6, 3] = 1
        q_table[7, 2] = 1
        print(q_table)"""

        #Method for nxm boards
        #Get the size of the map
        size_x = self.problem.building.size[0]
        size_y = self.problem.building.size[1]

        #Iterate through the states and find the best action for each state
        #Very error-prone. Gets lost in cycles easily
        for i, value in enumerate(self.states):

            pos_x = value[-2]
            pos_y = value[-1]

            possible_act = []

            #set action to clean if cell is dirty
            if not value[pos_x * size_y + pos_y]:
                q_table[i, 0] = 1
            #set action to rnadom movement if cell is clean
            else:
                if pos_x != size_x-1:
                    possible_act.append(3)
                if pos_y != size_y-1:
                    possible_act.append(1)
                if pos_x != 0:
                    possible_act.append(2)
                if pos_y != 0:
                    possible_act.append(4)
                q_table [i, np.random.choice(possible_act)] =1
                q_table [i, np.random.choice(possible_act)] =1
                print(possible_act)
        print(q_table)

        #print(self.actions)
        #print(self.states)
        #print(len(self.states))
        #print(self.problem.get_all_states())
        #print(self.problem.get_all_actions())
        # Put your source code here
        # Nothing to see here
        # w.g. q_table[0, 1] = 5 asserts a q_value of 5 to perform action 1 in state 0
        # the corresponding states and actions can be obtained by self.states[0] and
        # self.actions[1] in this example"""


        return q_table

