# dqn_gym.py
# Deep Q-Learning example adapted for Gymnasium (book-style DQN)
# Usage: python neural-network-example.py --env CartPole-v1
# Requires: torch, numpy, gymnasium

import argparse
import collections
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.uint8),
                np.array(next_states))


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        # If input is image-like (C,H,W) and H,W > 10 -> use conv stack similar to the book
        if len(input_shape) == 3 and input_shape[1] >= 16 and input_shape[2] >= 16:
            c, h, w = input_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            conv_out_size = self._get_conv_out(input_shape)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )
            self.is_conv = True
        else:
            # Flattened / low-dimensional observation: use simple MLP
            flattened_size = int(np.prod(input_shape))
            self.fc = nn.Sequential(
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )
            self.is_conv = False

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x expected shape: (batch, ...) where ... = input_shape
        if self.is_conv:
            conv_out = self.conv(x).view(x.size()[0], -1)
            return self.fc(conv_out)
        else:
            flat = x.view(x.size()[0], -1)
            return self.fc(flat)


def calc_loss(batch, net, tgt_net, device="cpu", gamma=0.99):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # get Q(s,a) for taken actions
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # compute next state values from target network
    next_state_values = tgt_net(next_states_v).max(1)[0]
    # zero-out for terminal states
    next_state_values[done_mask] = 0.0
    # detach so gradients do not flow into target network
    next_state_values = next_state_values.detach()

    expected_state_action_values = rewards_v + gamma * next_state_values
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    return loss


class Agent:
    def __init__(self, env, buffer):
        self.env = env
        self.exp_buffer = buffer
        self.state = None
        self.reset()

    def reset(self):
        obs, _ = self.env.reset()
        self.state = obs
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        # choose action
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.tensor(np.array([self.state]), dtype=torch.float32).to(device)
            q_vals = net(state_v)
            _, act_v = torch.max(q_vals, dim=1)
            action = int(act_v.item())

        # step
        new_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        exp = Experience(self.state, action, reward, done, new_state)
        self.exp_buffer.append(exp)
        self.total_reward += reward
        self.state = new_state

        if done:
            total = self.total_reward
            self.reset()
            return total
        return None


def make_env(env_name):
    # Basic wrapper: use gymnasium.make; user can add further wrappers externally if needed
    env = gym.make(env_name)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env name")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enable cuda")
    parser.add_argument("--mean_reward", type=float, default=195.0, help="Stop when mean reward > this")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    ENV_NAME = args.env
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    REPLAY_START_SIZE = 1000
    LEARNING_RATE = 1e-4
    SYNC_TARGET_FRAMES = 1000
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02
    EPSILON_DECAY_LAST_FRAME = 100000  # linear decay over frames

    env = make_env(ENV_NAME)
    test_env = make_env(ENV_NAME)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    net = DQN(obs_shape, n_actions).to(device)
    tgt_net = DQN(obs_shape, n_actions).to(device)
    tgt_net.load_state_dict(net.state_dict())

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    frame_idx = 0
    best_mean_reward = None
    ts_frame = 0
    ts = time.time()

    print(net)
    print("Observations:", obs_shape, "Actions:", n_actions)
    print("Device:", device)

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)

        if len(buffer) < REPLAY_START_SIZE:
            # wait until buffer is populated
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # sample batch and train
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device, gamma=GAMMA)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        if reward is not None:
            speed = (frame_idx - ts_frame) / (time.time() - ts + 1e-9)
            ts_frame = frame_idx
            ts = time.time()

            mean_reward = np.mean(total_rewards[-100:]) if total_rewards else 0.0
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" %
                  (frame_idx, len(total_rewards), mean_reward, epsilon, speed))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                print("Best mean reward updated %.3f -> %.3f, model saved" %
                      (best_mean_reward if best_mean_reward is not None else 0.0, mean_reward))
                best_mean_reward = mean_reward
                torch.save(net.state_dict(), ENV_NAME + "-best.dat")

            if mean_reward > args.mean_reward:
                print("Solved in %d frames!" % frame_idx)
                break


if __name__ == "__main__":
    main()
