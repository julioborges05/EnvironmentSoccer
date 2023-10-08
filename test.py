import gym
import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSS-v1')

env.reset()
print(env.action_space)
# Run for 1 episode and print reward at the end
for i in range(10000):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        # print(reward)
    print(reward)