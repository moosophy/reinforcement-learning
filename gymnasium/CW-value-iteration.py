import gymnasium as gym

#The overall logic of our code is simple: in the loop, we play 100 random steps from the environment, populating the
# reward and transition tables. After those 100 steps, we perform a value iteration loop over all states, updating our value
# table. Then we play several full episodes to check our improvements using the updated value table. If the average reward
# for those test episodes is above the 0.8 boundary , then we stop training. During test episodes, we also update our reward
# and transition tables to use all data from the environment.

GAMMA = 0.9
GOAL_STATE = 47
NUM_OF_ITERATIONS = 50


class Agent:
    def __init__(self):
        self.env = gym.make("CliffWalking-v1", is_slippery=True)
        self.state, _ = self.env.reset()

        self.reward_table = dict()
        self.transition_table = dict()
        self.value_table = dict()
    

    # Populates reward_table and transition_table
    def random_n_steps(self, n):
        for _ in range(n):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, info = self.env.step(action)

            if next_state == GOAL_STATE: reward = 100
            self.reward_table[(self.state, action, next_state)] = reward

            # making a nested dictionary transition_table
            if (self.state, action) not in self.transition_table:
                self.transition_table[(self.state, action)] = dict()            
            if next_state not in self.transition_table[(self.state, action)].keys():
                self.transition_table[(self.state, action)][next_state] = 1
            else:
                self.transition_table[(self.state, action)][next_state] += 1

            self.state = next_state
        
    
    # Calculates and returns action value Q_s_a from the formula (bellman equation), using reward_table
    # to get reward value and transition table to get the probabilities
    def calc_action_value(self, state, action):
        if (state, action) not in self.transition_table:
            return 0.0
    
        total_count = sum(self.transition_table[(state, action)].values())

        value = 0.0
        for next_state, count in self.transition_table[(state, action)].items():
            probability = count/total_count
            reward = self.reward_table[(state, action, next_state)]
            if next_state not in self.value_table:
                self.value_table[next_state] = 0
            value += probability*(reward+GAMMA*self.value_table[next_state])

        return value
    
    # Given a state, it calculates the values of all possible actions from that state using calc_action_value() 
    # and selects the best one
    def select_best_action(self, state):
        best_value, best_action = -1000000.0, 0
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if action_value >= best_value:
                best_value = action_value
                best_action = action
        return best_action
    

    # One full episode that lasts until terminated or truncated. still populates reward_table and transition_table
    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()
        terminated, truncated = False, False
        
        turn = 0
        while True:
            action = self.select_best_action(state)
            # print(f"chose action {action} on state {state}")
            next_state, reward, terminated, truncated, info = env.step(action)

            if next_state == GOAL_STATE: reward = 100
            self.reward_table[(state, action, next_state)] = reward
            total_reward += reward

            if (state, action) not in self.transition_table:
                self.transition_table[(state, action)] = dict()            
            if next_state not in self.transition_table[(state, action)].keys():
                self.transition_table[(state, action)][next_state] = 1
            else:
                self.transition_table[(state, action)][next_state] += 1

            if terminated or truncated:
                break

            state = next_state
            turn+=1

        return total_reward

    
    # for every state we see the best action to take, thus calculating the value of 
    # each state and populating the value_table
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            action_values = []
            for action in range(self.env.action_space.n):
                value = self.calc_action_value(state, action)
                action_values.append(value)
            
            self.value_table[state] = max(action_values)



if __name__ == "__main__":
    env = gym.make("CliffWalking-v1", is_slippery=True)
    env = gym.wrappers.TimeLimit(env, 200)
    agent = Agent()

    best_reward = 0.0
    iteration = 0

    while iteration < NUM_OF_ITERATIONS:
        iteration += 1

        # collect some random transitions
        agent.random_n_steps(10000)
        


        # # For debugging
        # print("\nReward table:")
        # for key, value in agent.reward_table.items():
        #     if key[0] == 35:
        #         print(key, value)
        # print("\nTransition table:")
        # for key, value in agent.transition_table.items():
        #     if key[0] == 35:
        #         print(key, value)
        # print("Action value for moving left from state 35:")     
        # print(agent.calc_action_value(35,3))
        # print("Up:")     
        # print(agent.calc_action_value(35,0))
        # print("Right:")     
        # print(agent.calc_action_value(35,1))
        # print("Down:")     
        # print(agent.calc_action_value(35,2))
        # #results:
        #     # Action value for moving left from state 35:
        #     # 218.9182225610088
        #     # Up:
        #     # 171.90144978977065
        #     # Right:
        #     # 256.3212254783902
        #     # Down:
        #     # 259.8017050675947


        # update value function (one sweep)
        
        agent.value_iteration()

        # # for debugging
        # print("value iteration done")
        # print("value_table:")
        # for state, value in agent.value_table.items():
        #     print(state, value)


        # test current policy
        reward = 0.0
        for _ in range(20):
            reward += agent.play_episode(env)
        reward /= 20

        print(f"Iteration {iteration}, average reward = {reward:.3f}")
    

    # check with visuals
    env = gym.make("CliffWalking-v1", is_slippery=True, render_mode="human")
    env = gym.wrappers.TimeLimit(env, 200)
    agent.play_episode(env)





