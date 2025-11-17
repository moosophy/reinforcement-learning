import gymnasium as gym

#The overall logic of our code is simple: in the loop, we play 100 random steps from the environment, populating the
# reward and transition tables. After those 100 steps, we perform a value iteration loop over all states, updating our value
# table. Then we play several full episodes to check our improvements using the updated value table. If the average reward
# for those test episodes is above the 0.8 boundary , then we stop training. During test episodes, we also update our reward
# and transition tables to use all data from the environment.

class Agent:
    def __init__(self):
        self.env = gym.make("CliffWalking-v1", render_mode="human", is_slippery=True)
        self.state, _ = self.env.reset()

        self.reward_table = dict()
        self.transition_table = dict()
    
    def random_n_steps(self, n):
        for _ in range(n):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.reward_table[(self.state, action, next_state)] = reward

            # making a nested dictionary transition_table

            if (self.state, action) not in self.transition_table:
                self.transition_table[(self.state, action)] = dict()            
            if next_state not in self.transition_table[(self.state, action)].keys():
                self.transition_table[(self.state, action)][next_state] = 1
            else:
                self.transition_table[(self.state, action)][next_state] += 1

            self.state = next_state


if __name__ == "__main__":
    agent = Agent()
    agent.random_n_steps(20)

    print("\nReward table:")
    for key, value in agent.reward_table.items():
        print(key, value)

    print("\nTransition table:")
    for key, value in agent.transition_table.items():
        print(key, value)



