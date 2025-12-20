
from turtle import reset
import gymnasium as gym
import numpy as np
from sympy import li
import torch
import torch.nn as nn
import torch.optim as optim


EPS = 2000
BATCH = 32

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(),            #input:
            nn.ReLU(),
            nn.Linear()             #output:
        )
    
    def forward(self, input):
        return(self.fc(input))
    


class Agent():
    def __init__(self):
        self.env = gym.make("CartPole-v1")


    def select_action(self, state):
        return action
    

    def play_episode(self, env=None):
        if env == None:
            env = self.env

        state, _ = self.env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        
        while not (terminated or truncated):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step()

            self.select_action(state)

            state = next_state


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent()
    
    for i in range(BATCH, EPS, BATCH):
        agent.play_episode()





        


