import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

#it plateaus at 200 - it decides that just standing in one place is better (T-T) 
#takes 1000 episodes to reach the plateau

GAMMA = 0.99
LR = 0.001

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # we use embedding because the states are descrete
        self.embedding = nn.Embedding(500, 100) # 500 - number of states
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 6) # 6 - number of actions
        )

    def forward(self, state):
        x = self.embedding(state)
        return self.net(x)  


class Agent:
    def __init__(self):
        env = gym.make("Taxi-v3")
        self.env = gym.wrappers.TimeLimit(env, 200)

        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        self.saved_log_probs = []
        self.rewards = []


    def select_action(self, state):
        state_t = torch.tensor([state], dtype=torch.long)

        logits = self.policy(state_t)
        probs = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        self.saved_log_probs.append(log_prob)

        return action.item()


    def play_episode(self, env=None, no_learning=False):
        if env is None:
            env = self.env

        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        if not no_learning:
            self.saved_log_probs = []
            self.rewards = []

        while not (terminated or truncated):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if not no_learning:
                self.rewards.append(reward)

            state = next_state

        # only updating if we are training (not testing/visualization)
        if not no_learning:
            self.update_policy()

        return total_reward


    def update_policy(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()

        # policy gradient loss = -sum(log(pi(a|s)) * G_t)
        loss = 0
        for log_prob, G_t in zip(self.saved_log_probs, returns):
            loss += -log_prob * G_t
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")
    env = gym.wrappers.TimeLimit(env, 200)
    agent = Agent()

    for i in range(50):
        # train 100 episodes
        for _ in range(100):
            agent.play_episode()

        # evaluate for 10 episodes
        reward = 0
        for _ in range(10):
            reward += agent.play_episode(env=None, no_learning=True)

        reward /= 10
        print(f"Average reward after {i*100} episodes: {reward}")

    # for visualization in the end:
    agent.play_episode(env, no_learning=True)
