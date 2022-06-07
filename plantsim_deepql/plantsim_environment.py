from pyparsing import actions

from ..plantsim.plantsim import Plantsim
from plantsim.pandas_table import PandasTable
from ..VC_World_RL_2.problem import Problem

class Environment:

    def __init__(self):

        self.plantsim
        pass

    def reset(self):
        pass

    def GetCurrentState(self):
        #update the table CurrentState
        #get the table CurrentState
        pass

class PS(Problem):

    def __init__(self, plantsim: Plantsim):
        self.plantsim = plantsim
        self.actions = PandasTable(plantsim=plantsim, object_name='Actions').table.set_index(keys='Index')
        self.states = PandasTable(plantsim=plantsim, object_name='States').table.set_index(keys='Index')

    def copy(self):
        """
        Creates a deep copy of itself
        :return: Problem
        """
        return copy.deepcopy(self)

    def act(self, action):
        """
        Peforms the action passed
        """
        pass

    def to_state(self, state):
        """
        Creates a tuple of the relevant state attributes
        :return: tuple()
        """
        state_tuple = (state["id"], state["color"], state["location"], state["evaluation"], state["goal_state"])
        return state_tuple

    def is_goal_state(self, state):
        """
        Checks if state is a goal state
        :param state: Problem
        :return: Boolean
        """
        if state["goal_state"] is True:
            return True
        else:
            return False

    def get_applicable_actions(self, state):
        """
        Returns a list of actions applicable in state
        :param state: Problem
        :return: list<String>
        """
        if state["location"] == "Verteiler":
            return ["move_to_station_1", "move_to_station_2"]
        elif state["location"] == "Senke":
            return []
        else:
            return ["go_on"]


    def get_current_state(self):
        """
        returns itself and eventually performs an update first
        :return:
        """
        return self.plantsim.get_current_state()

    def eval(self, state):
        """
        Evaluates the state
        :param state: Problem
        :return: float
        """
        return state["evaluation"]

    def get_all_actions(self):
        """
        returns a list of all actions
        :return: list<string>
        """
        return self.actions

    def get_all_states(self):
        """
        returns a list of all states
        :return: list<string>
        """
        return self.states

    def get_reward(self, state):
        """
        Calulates a reward of the state for RL
        :param state: Problem
        :return: float
        """
        return -self.eval(state)

    def reset(self):
        """
        resets the environment
        """
        self.plantsim.reset_simulation()