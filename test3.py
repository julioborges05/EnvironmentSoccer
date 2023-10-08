import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Crie o ambiente
env = gym.make("VSS-v1")

# Parâmetros
state_size = env.observation_space.shape[0]
action_size = 2
learning_rate = 0.001
gamma = 0.99  # Fator de desconto para recompensas futuras

# Modelo da rede neural
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# Função para escolher ação com base em uma política epsilon-greedy
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Ação aleatória
    else:
        q_values = model.predict(state)
        return np.argmax(q_values[0])  # Ação com maior valor Q

# Treinamento
num_episodes = 500
epsilon = 1.0  # Taxa de exploração inicial
epsilon_min = 0.01
epsilon_decay = 0.995

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):  # Limite máximo de etapas por episódio
        env.render()

        action = choose_action(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Atualize o valor Q usando a fórmula Q-learning
        target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target

        # Treine o modelo com o novo valor alvo
        model.fit(state, target_f, epochs=1, verbose=0)

        total_reward += reward
        state = next_state

        if done:
            break

    # Reduza a taxa de exploração epsilon com o tempo
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episódio {episode + 1}: Recompensa Total = {total_reward}")

# Feche o ambiente
env.close()
