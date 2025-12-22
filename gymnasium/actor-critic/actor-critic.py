
import gymnasium as gym
import numpy as np
from sympy import li
import torch
import torch.nn as nn
import torch.optim as optim


#This version is much better, but still never stabilizes completely.

EPS = 10000
ROLLOUT = 32        #num of steps to take before using them to update the network
GAMMA = 0.9
ALPHA = 0.5
LEARNING_RATE = 0.0001


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_shape, 128),           
            nn.ReLU(),       
        )
        self.actor_head = nn.Linear(128, num_actions)
        self.critic_head = nn.Linear(128, 1)
    
    def forward(self, input):
        input = self.body(input)
        actor_logits = self.actor_head(input)
        critic_V = self.critic_head(input)
        return actor_logits, critic_V
    



class Agent():
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.net = ActorCritic(int(np.prod(self.env.observation_space.shape)), self.env.action_space.n)

        self.softmax = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        self.ep_number = 0
        self.states, self.actions, self.rewards = [], [], []



    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.net(state)
        probabilities = self.softmax(logits)
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action
    


    def update_network(self, is_terminated):
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        logits, Vs = self.net(states)
        Vs = Vs.squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)                 #we calculate only for the action we actually took

        # print("\nlog_probs tensor")
        # for i in range (3): (print(log_probs[i])   )    

        if is_terminated:
            R = 0
        else:
            R = Vs[-1].detach()

        
        total_loss = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R
            advantage = R - Vs[i]
            loss_actor = -log_probs[i] * advantage.detach()
            loss_critic = advantage**2
            total_loss += loss_actor + ALPHA * loss_critic

        entropy = dist.entropy()
        loss_entropy = -0.01 * entropy.sum()
        total_loss += loss_entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.states.clear()
        self.rewards.clear()
        self.actions.clear()

    

    def play_episode(self, env=None):
        if env == None:
            env = self.env

        state, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        step = 0
        
        while not (terminated or truncated):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward+= reward

            state = torch.tensor(state)

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

            step += 1
            if step >= ROLLOUT or (terminated or truncated):
                self.update_network(terminated or truncated)
                step = 0

            state = next_state
        
        return total_reward
    


    def evaluate(self, env=None):
        if env == None:
            env = self.env

        state, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0

        with torch.no_grad():
            while not (terminated or truncated):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                logits, _ = self.net(state)
                probabilities = self.softmax(logits)

                action = torch.argmax(probabilities, dim=1).item()
                next_state, reward, terminated, truncated, _ = env.step(action)

                total_reward+= reward
                state = next_state

        return total_reward




if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent()
    
    for i in range(EPS):
        agent.play_episode()
        if not i % 100:
            reward = agent.evaluate()
            print(f"total reward after {i} episodes: {reward}")
    
    #Visualization:
    agent.evaluate(env=env)





        


