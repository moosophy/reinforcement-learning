import gymnasium as gym

# The same logic as in GYM-CW-value-iteration.py, but we combined calc_action value with value_iteration. This way
# our value_table transforms into action_value_table so we only store Qs, and we calculate the values of states
# based on the Qs.

GAMMA = 0.9
GOAL_STATE = 47
NUM_OF_ITERATIONS = 50


class Agent:
    def __init__(self):
        self.env = gym.make("CliffWalking-v1", is_slippery=True)
        self.state, _ = self.env.reset()

        self.reward_table = dict()
        self.transition_table = dict()
        self.action_value_table = dict()
    

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
    
    # Given a state, it takes the values of all possible acitons from action_value_table, selects the one with
    # the best value, and returns the action.
    def select_best_action(self, state):
        best_value, best_action = -1000000.0, 0
        for action in range(self.env.action_space.n):
                
            action_value = self.action_value_table.get((state, action), 0)
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

    
    # Calculates the action value of every action of every state using Bellman Equation. And adds this value to the 
    # action_value_table.
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):

                action_value = 0.0
                # total_count = sum(self.transition_table[(state, action)].values() if (state, action) in self.transition_table)
                total_count = sum(self.transition_table.get((state, action), {}).values())

                for next_state, count in self.transition_table.get((state, action), {}).items():

                    probability = count/total_count
                    reward = self.reward_table[(state, action, next_state)]
                    next_state_value = self.action_value_table.get((next_state, self.select_best_action(next_state)), 0)

                    # Bellman equation
                    action_value += probability*(reward+GAMMA*next_state_value)
            
                self.action_value_table[state, action] = action_value



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





