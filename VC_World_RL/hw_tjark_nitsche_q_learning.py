def train(self):
    s_act = self.problem.get_current_state()
    act_index = self.states.index(s_act.to_state())
    # r = self.problem.get_reward(s1)
    alpha = 0.3
    epsilon = 0.3

    # print (s1, r)
    # q_table = ()
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

        q_value_new = q + alpha * (r + self.gamma * new_Q_max - q)
        self.q_table[act_index][self.actions.index(action)] = q_value_new

        s_act = s_new
        act_index = self.states.index(s_act.to_state())

    return self.q_table, N_sa