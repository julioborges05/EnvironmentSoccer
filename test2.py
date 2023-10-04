import gym
import numpy as np
import tensorflow as tf

# Crie o ambiente CartPole
env = gym.make('CartPole-v1')

# Defina a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])

# Defina o otimizador e a função de perda
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.mean_squared_error

# Hiperparâmetros
gamma = 0.99  # Fator de desconto
epsilon = 0.1  # Taxa de exploração

# Função para escolher ação com base na política epsilon-greedy
def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Ação aleatória
    else:
        q_values = model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])

# Loop de treinamento
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)

        # Calcula o alvo Q
        q_values = model.predict(state.reshape(1, -1))
        next_q_values = model.predict(next_state.reshape(1, -1))
        target_q = q_values.copy()
        target_q[0][action] = reward + gamma * np.max(next_q_values)

        # Treina a rede neural
        with tf.GradientTape() as tape:
            predicted_q = model(state.reshape(1, -1))
            loss = loss_fn(target_q, predicted_q)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_reward += reward
        state = next_state

    print(f"Episódio {episode + 1}: Recompensa total = {total_reward}")

# Avaliação do agente treinado
num_eval_episodes = 10
eval_rewards = []

for _ in range(num_eval_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = np.argmax(model.predict(state.reshape(1, -1)))
        state, reward, done, _ = env.step(action)
        total_reward += reward

    eval_rewards.append(total_reward)

average_eval_reward = np.mean(eval_rewards)
print(f"Recompensa média em {num_eval_episodes} episódios de teste: {average_eval_reward}")
