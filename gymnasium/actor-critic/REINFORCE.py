import gymnasium as gym
import numpy as np
from sympy import li
import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPS = 4000                  #number of episodes to play
BATCH_NUM = 50




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
        
        self.all_log_probs = []
        self.rewards = [[] for _ in range(BATCH_NUM)]         
        self.episode_number = 0     # goes up to BATCH_NUM, then resets (for batch management)



    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        state_t = state_t.unsqueeze(0)

        prediction = self.policy(state_t)
        probabilities = self.softmax(prediction)            #for example [0.7, 0.3], so 70% left, 30% right.
        action = torch.multinomial(probabilities, num_samples=1).item()     #picks the action based on its probability

        log_prob = torch.log(probabilities[0, action])
        self.all_log_probs.append(log_prob)

        return action
    


    def update_policy(self):
        #self.episode_number = 20, 20 elements in self.log_probabilities and self.rewards
        
        all_returns = []
        for ep in range(BATCH_NUM):
            ep_returns = []
            G = 0
            for r in reversed(self.rewards[ep]):    #going backwards
                G = r + GAMMA * G
                ep_returns.append(G)

            ep_returns.reverse()
            all_returns.extend(ep_returns)
        
        #all_returns and all_log_probs have the same length
        
        all_returns = torch.tensor(all_returns, dtype=torch.float32)

        # all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
        
        loss = 0
        for i in range(len(self.all_log_probs)):
            loss += -self.all_log_probs[i] * all_returns[i]
        
        #normalize the loss so that it is not batch dependant (goes from 3763.1211 to 0.0137)
        loss = loss / len(self.all_log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        self.all_log_probs.clear()
        self.rewards = [[] for _ in range(BATCH_NUM)]  

        
    
    def play_episode(self, env=None):
        if env == None:
            env = self.env
        state, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0

        while not (terminated or truncated):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            self.rewards[self.episode_number].append(reward)

            state = next_state
        
        self.episode_number += 1
        
        if self.episode_number == BATCH_NUM:
            self.update_policy()
            self.episode_number = 0
        return total_reward
    


    def evaluate(self, env=None):
        if env == None:
            env = self.env
        state, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0

        with torch.no_grad():
            while not (terminated or truncated):
                state_t = torch.tensor(state, dtype=torch.float32)
                state_t = state_t.unsqueeze(0)

                prediction = self.policy(state_t)
                probabilities = self.softmax(prediction) 
                
                action = torch.argmax(probabilities, dim=1).item() #picks the best action

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

        return total_reward





if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent()

    for i in range(BATCH_NUM, EPS, BATCH_NUM):
        # train one batch
        for _ in range(BATCH_NUM):
            agent.play_episode()

        # evaluate for 5 episodes
        reward = 0
        for _ in range(5):
            reward += agent.evaluate()
        reward /= 5

        print(f"Average reward after {i} episodes: {reward}")

    
    # final visualization
    agent.evaluate(env=env)





