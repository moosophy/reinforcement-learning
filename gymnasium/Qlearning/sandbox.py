#the file has "python analysis ignore" setting turned on - there are no markers indicating incorrect syntax

class Agent:
    def __init__(self):
        self.Q_table = dict()           #[(state, action)] : action_value


    def select_action(self, state, selected_epsilon):
        # Given a state, it takes the values of all possible acitons from Q_table, selects the one with
        # the best value, and returns the action. There is also an epsilon chance that the action will be 
        # random (the agent explores)
    
    def play_episode(self):
        # play an episode of the environment, at each step we collect the state, action, reward, and new state.
        # We then use the Q-learning update using all the collected values to update this action value.

