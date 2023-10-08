import gym
import rsoccer_gym
from tensorflow import keras

env = gym.make('VSS-v1')
observation = env.reset()

num_episodes = 10000


model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)



for _ in range (num_episodes):
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
    print(reward)

env.close()
