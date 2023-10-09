import gym
import rsoccer_gym

env = gym.make('VSS-v1')
observation = env.reset()

num_episodes = 5

for _ in range(num_episodes):
    observation = env.reset()
    done = False
    total_rewords = 0

    while not done:
        #Escolha da ação
        action = env.action_space.sample() #Substituir pela estrategia escolhida

        #Ação no ambiente
        next_state, reward, done, _ = env.step(action)

        total_rewords += reward

        env.render()
        print("Premio atual: ", reward)
    print("Soma dos premios: ", total_rewords)

env.close()
