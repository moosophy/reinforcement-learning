import gymnasium as gym
import random
# TODO: separate the exploration from exploitation like in the

GAMMA = 0.9
ALPHA = 0.2             # α - how much a new sample contributes to the action value (blending)
GOAL_STATE = 47

class Agent:
    def __init__(self):
        env = gym.make("CliffWalking-v1", is_slippery=True)
        self.env = gym.wrappers.TimeLimit(env, 200)

        self.Q_table = dict()           #[(state, action)] : action_value


    def select_action(self, state, selected_epsilon):
        epsilon = random.random()
        if epsilon < selected_epsilon:
            return self.env.action_space.sample()

        best_action, best_value = 0, -10000
        for action in range(self.env.action_space.n):
            action_value = self.Q_table.get((state, action), 0)
            if action_value >= best_value:
                best_value = action_value
                best_action = action

        return best_action


    def play_episode(self, env=None, epsilon=0):
        if env == None:
            env = self.env

        state, _ = env.reset()
        truncated, terminated = False, False
        total_reward = 0.0

        while not (truncated or terminated):
            action = self.select_action(state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            if next_state == GOAL_STATE: reward = 100

            total_reward += reward

            if terminated or truncated:
                best_next_Q = 0                                                                              #need to think this over
            else:
                best_next_Q = max( self.Q_table.get((next_state, act), 0) for act in range(env.action_space.n) )

            new_Q = reward + GAMMA * best_next_Q
            old_Q = self.Q_table.get((state, action), 0)
            self.Q_table[(state, action)] = old_Q + ALPHA * (new_Q - old_Q)

            state = next_state

        return total_reward
    

if __name__ == "__main__":
    env = gym.make("CliffWalking-v1", is_slippery=True, render_mode="human")
    env = gym.wrappers.TimeLimit(env, 200)
    agent = Agent()

    
    epsilon = 0.5
    for i in range(10):
        #first we train our agent by playing 10 episodes with high exploration rate
        for _ in range (10):            
            agent.play_episode(epsilon=epsilon)
    
        #then we test it with high exploitation rate
        reward = 0
        for _ in range(10):
            reward += agent.play_episode()
        reward /= 10
        print(f"Average reward after playing {i*10} episodes: {reward}")
    

    # For visualization:
    agent.play_episode(env, epsilon=0)
            




