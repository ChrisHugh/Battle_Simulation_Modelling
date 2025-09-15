import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gymnasium as gym

from battle_env_gym import BattleArenaEnv


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


def build_q_network(observation_size: int, num_actions: int) -> keras.Model:
    inputs = keras.Input(shape=(observation_size,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    environment = BattleArenaEnv(max_steps=600, role="archer", headless=False)
    num_actions = environment.action_space.n
    observation, _ = environment.reset()
    observation_size = int(np.prod(observation.shape))

    online_q_network = build_q_network(observation_size, num_actions)
    target_q_network = build_q_network(observation_size, num_actions)
    target_q_network.set_weights(online_q_network.get_weights())
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_function = keras.losses.Huber()

    replay_buffer = ReplayBuffer(capacity=100000)
    discount_gamma = 0.99
    batch_size = 128
    exploration_epsilon = 1.0
    exploration_epsilon_min = 0.05
    exploration_epsilon_decay = 0.995
    target_update_frequency = 1000
    total_training_steps = 10000

    # Visualization controls
    render_during_training = True  # set to False to disable
    render_every_n_steps = 1       # increase to reduce overhead (e.g., 5 or 10)

    state = observation
    episode_reward = 0.0
    for step in range(1, total_training_steps + 1):
        if np.random.rand() < exploration_epsilon:
            action = environment.action_space.sample()
        else:
            q_values = online_q_network.predict(state[np.newaxis, :], verbose=0)[0]
            action = int(np.argmax(q_values))

        next_state, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated

        if render_during_training and (step % render_every_n_steps == 0):
            # Provide overlay text for interpretability during training
            environment.overlay_text = (
                f"Training step: {step}  epsilon: {exploration_epsilon:.2f}  "
                f"episode_reward: {episode_reward:.2f}  role: {environment.role}"
            )
            environment.render(mode="human")

        replay_buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward

        if done:
            state, _ = environment.reset()
            episode_reward = 0.0

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            next_state_q_values = target_q_network.predict(next_states, verbose=0)
            max_next_state_q = np.max(next_state_q_values, axis=1)
            td_targets = rewards + discount_gamma * (1.0 - dones) * max_next_state_q

            with tf.GradientTape() as tape:
                current_q_values = online_q_network(states, training=True)
                action_masks = tf.one_hot(actions.astype(int), num_actions)
                selected_q_values = tf.reduce_sum(current_q_values * action_masks, axis=1)
                loss = loss_function(td_targets, selected_q_values)
            gradients = tape.gradient(loss, online_q_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, online_q_network.trainable_variables))

        if step % target_update_frequency == 0:
            target_q_network.set_weights(online_q_network.get_weights())

        exploration_epsilon = max(exploration_epsilon_min, exploration_epsilon * exploration_epsilon_decay)

    os.makedirs("models", exist_ok=True)
    online_q_network.save("models/dqn_tf_model.keras")


if __name__ == "__main__":
    main()


