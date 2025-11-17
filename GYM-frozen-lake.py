import gymnasium as gym
import collections

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, is_slippery=True)
        self.state, _ = self.env.reset()

        # reward[(s, a, s2)] = reward_value
        self.rewards = collections.defaultdict(float)

        # transits[(s, a)][s2] = count
        self.transits = collections.defaultdict(collections.Counter)

        # values[s] = state_value
        self.values = collections.defaultdict(float)

    # ---- EXPERIENCE COLLECTION ----
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, _ = self.env.step(action)

            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1

            if terminated or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    # ---- ACTION VALUE CALCULATION (Q-value) ----
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())

        # If we have no data for this (state, action), return 0
        if total == 0:
            return 0.0

        value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            prob = count / total
            value += prob * (reward + GAMMA * self.values[tgt_state])

        return value

    # ---- GREEDY ACTION SELECTION ----
    def select_action(self, state):
        best_action = None
        best_value = None

        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action

    # ---- PLAY ONE EPISODE USING CURRENT VALUE ESTIMATE ----
    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()

        while True:
            action = self.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            # update tables with this real experience
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1

            total_reward += reward

            if terminated or truncated:
                break

            state = new_state

        return total_reward

    # ---- VALUE ITERATION UPDATE ----
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            action_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(action_values)


# ============================
#        TRAINING LOOP
# ============================

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME, is_slippery=True)
    agent = Agent()

    best_reward = 0.0
    iteration = 0

    while True:
        iteration += 1

        # collect some random transitions
        agent.play_n_random_steps(100)

        # update value function (one sweep)
        agent.value_iteration()

        # test current policy
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        print(f"Iteration {iteration}, average reward = {reward:.3f}")

        if reward > best_reward:
            print(f"  best reward improved: {best_reward:.3f} -> {reward:.3f}")
            best_reward = reward

        # stop condition
        if reward > 0.80:
            print(f"Solved in {iteration} iterations!")
            break
