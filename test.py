import gym
import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSS-v1')

env.reset()

total_reward = 0
# Run for 1 episode and print reward at the end
for i in range(100):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward
        print("Premio atual: ", reward)
    print("Soma dos premios: ", total_reward)
