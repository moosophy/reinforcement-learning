import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01




class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            #observation space: Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)
            nn.Linear(4, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) #two actions: left, right.
        )
    
    def forward(self, input):
        return self.fc(input)




class Agent:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.policy = NeuralNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

        self.softmax = nn.Softmax(dim=1)
        
        self.log_probabilities = [] #for calculating the loss
        self.rewards = []



    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        state_t = state_t.unsqueeze(0)

        prediction = self.policy(state_t)
        probabilities = self.softmax(prediction)            #for example [0.7, 0.3], so 70% left, 30% right.
        action = torch.multinomial(probabilities, num_samples=1).item()     #picks the action based on its probability

        log_prob = torch.log(probabilities[0, action])
        self.log_probabilities.append(log_prob)

        return action
    


    def update_policy(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):    #going backwards
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32)

        loss = 0
        for i in range(len(self.log_probabilities)):
            loss += -self.log_probabilities[i] * returns[i]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probabilities.clear()
        self.rewards.clear()

        
    
    def play_episode(self, env=None):
        state, _ = self.env.reset()
        terminated, truncated = False, False
        total_reward = 0.0

        while not (terminated or truncated):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            self.rewards.append(reward)

            state = next_state
        
        self.update_policy()
        return total_reward




if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent()

    for i in range(1, 100):
        # train 10 episodes per batch
        for _ in range(10):
            agent.play_episode()

        # evaluate for 5 episodes
        reward = 0
        for _ in range(5):
            reward += agent.play_episode()
        reward /= 5

        print(f"Average reward after {i*10} episodes: {reward}")

        if reward ==500.0:
            agent.play_episode(env= env)
    
    # final visualization
    agent.play_episode(env= env)





