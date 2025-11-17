import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset()  
truncated, terminated = False, False

while not (truncated or terminated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()