#the file has "python analysis ignore" setting turned on - there are no markers indicating incorrect syntax

class Agent:
    def __init__(self):
        self.reward_table = dict()
        self.transition_table = dict()
        self.action_value_table = dict()

    def random_n_steps(self, n):
        # Simulates the environment, taking random steps, and uses the discovered data to populate self.reward_table and 
        # self.transition_table 

    def select_best_action(self, state):
        # Given a state, it calculates the values of all possible actions from that state using calc_action_value() and 
        # selects the best action (with the highest action value).
        return best_action
    
    def value_iteration(self):
        # Calculates the action value of every action of every state using Bellman Equation. And adds this value to the 
        # action_value_table. It uses reward_table to get reward value, transition table to get the probabilities, and 
        # state values using max(action_value_table[next_state, next_best_action])

#thus select_best_action() is our policy.