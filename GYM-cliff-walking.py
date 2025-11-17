import gymnasium as gym

#The overall logic of our code is simple: in the loop, we play 100 random steps from the environment, populating the
# reward and transition tables. After those 100 steps, we perform a value iteration loop over all states, updating our value
# table. Then we play several full episodes to check our improvements using the updated value table. If the average reward
# for those test episodes is above the 0.8 boundary , then we stop training. During test episodes, we also update our reward
# and transition tables to use all data from the environment.

GAMMA = 0.9


class Agent:
    def __init__(self):
        self.env = gym.make("CliffWalking-v1", render_mode="human", is_slippery=True)
        self.state, _ = self.env.reset()

        self.reward_table = dict()
        self.transition_table = dict()
        self.value_table = dict()
    
    def random_n_steps(self, n):
        for _ in range(n):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.reward_table[(self.state, action, next_state)] = reward
            print(reward)

            # making a nested dictionary transition_table

            if (self.state, action) not in self.transition_table:
                self.transition_table[(self.state, action)] = dict()            
            if next_state not in self.transition_table[(self.state, action)].keys():
                self.transition_table[(self.state, action)][next_state] = 1
            else:
                self.transition_table[(self.state, action)][next_state] += 1

            self.state = next_state
    
    #calculates and returns action value Q_s_a from the formula (bellman equation)
        #example
        #Transition table:
        # (36, 1) {36: 1, 24: 2}
        #Reward table:
        # (36, 1, 36) -100
        # (36, 1, 24) -1
    def calc_action_value(self, state, action):
        if (state, action) not in self.transition_table:
            return 0.0
    
        total_count = sum(self.transition_table[(state, action)].values())

        value = 0.0
        for next_state, count in self.transition_table[(state, action)].items():
            probability = count/total_count
            reward = self.reward_table[(state, action, next_state)]
            # Correct version:
            # value += probability*(reward+GAMMA*self.value_table[next_state])
            value += probability*(reward)

        return value
    
    #given a state, it calculates the values of all possible actions from that states and 
    # selects the best one
    def select_best_action(self, state):
        best_value, best_action = 0.0, 0
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if action_value >= best_value:
                best_value = action_value
                best_action = action
        return best_action


            


if __name__ == "__main__":
    agent = Agent()
    agent.random_n_steps(30)

    print("\nReward table:")
    for key, value in agent.reward_table.items():
        print(key, value)

    print("\nTransition table:")
    for key, value in agent.transition_table.items():
        print(key, value)

    print("\nAction value Q (initial state, move right):")
    print(agent.calc_action_value(36, 1))



