import os
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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


class BattleAgentNeuralNetwork(nn.Module):
    """
    Deep Q-Network for battle simulation agent.
    
    This network learns to predict the value of each possible action
    given the current game state (observation).
    
    Architecture:
    - Input: 12-dimensional observation vector (agent health, position, boss info, etc.)
    - Hidden Layer 1: 128 neurons with ReLU activation
    - Hidden Layer 2: 128 neurons with ReLU activation  
    - Output: 10 Q-values (one for each possible action: no-op + 8 movement directions)
    """
    
    def __init__(self, observation_vector_size: int, number_of_possible_actions: int):
        super().__init__()
        
        # First hidden layer: transforms observation into higher-level features
        self.observation_to_features_layer = nn.Linear(observation_vector_size, 128)
        
        # Second hidden layer: processes features to understand game state
        self.feature_processing_layer = nn.Linear(128, 128)
        
        # Output layer: predicts Q-value for each possible action
        self.action_value_prediction_layer = nn.Linear(128, number_of_possible_actions)

    def forward(self, observation_tensor):
        """
        Forward pass through the network.
        
        Args:
            observation_tensor: Current game state as a tensor
            
        Returns:
            action_values: Q-values for each possible action

        """
        # Three layers to observe current state, extract meaningful features, and produce q-values for available actions.
        hidden_features_1 = torch.relu(self.observation_to_features_layer(observation_tensor))
        hidden_features_2 = torch.relu(self.feature_processing_layer(hidden_features_1))
        action_values = self.action_value_prediction_layer(hidden_features_2)
        
        return action_values


def main():
    # Set device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # If using CUDA, you can set specific GPU and memory management
    if device.type == "cuda":
        torch.cuda.set_device(0)  # Use first GPU
        torch.cuda.empty_cache()  # Clear cache
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    environment = BattleArenaEnv(max_steps=1000, role="archer", headless=True)
    num_actions = environment.action_space.n
    observation, _ = environment.reset()
    observation_size = int(np.prod(observation.shape))

    # Create neural networks and move to device
    main_agent_brain = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    target_agent_brain = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    target_agent_brain.load_state_dict(main_agent_brain.state_dict())
    
    # Set up training optimizer and loss function
    neural_network_optimizer = optim.Adam(main_agent_brain.parameters(), lr=1e-3)
    prediction_error_function = nn.SmoothL1Loss()  # Equivalent to Huber loss

    # Experience replay buffer for storing past game experiences
    experience_memory = ReplayBuffer(capacity=100000)
    
    # Training hyperparameters
    future_reward_discount_factor = 0.999
    training_batch_size = 1024
    exploration_randomness = 1.0
    minimum_exploration_randomness = 0.05
    exploration_decay_rate = 0.9995
    target_network_update_frequency = 1000
    total_training_steps = 100000

    # Lightweight progress logging (no rendering required)
    log_every_n_steps = 1024
    recent_episode_rewards = deque(maxlen=20)
    last_training_loss_value = None

    # Visualization controls
    render_during_training = False  # set to False to disable
    render_every_n_steps = 1    # increase to reduce overhead (e.g., 5 or 10)

    current_game_state = observation
    current_episode_total_reward = 0.0
    
    for training_step in range(1, total_training_steps + 1):
        # Decide whether to explore (random action) or exploit (use neural network)
        if np.random.rand() < exploration_randomness:
            chosen_action = environment.action_space.sample()
            # Gradually reduce exploration randomness
            exploration_randomness = max(minimum_exploration_randomness, exploration_randomness * exploration_decay_rate)
        else:
            # Convert game state to tensor and move to device
            game_state_tensor = torch.FloatTensor(current_game_state).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_action_values = main_agent_brain(game_state_tensor).cpu().numpy()[0]
            chosen_action = int(np.argmax(predicted_action_values))

        next_game_state, immediate_reward, episode_terminated, episode_truncated, _ = environment.step(chosen_action)
        episode_finished = episode_terminated or episode_truncated

        if render_during_training and (training_step % render_every_n_steps == 0):
            # Provide overlay text for interpretability during training
            environment.overlay_text = (
                f"Training step: {training_step}  exploration: {exploration_randomness:.2f}  "
                f"episode_reward: {current_episode_total_reward:.2f}  role: {environment.role}  "
                f"device: {device}"
            )
            environment.render(mode="human")

        # Store experience in memory for later learning
        experience_memory.add((current_game_state, chosen_action, immediate_reward, next_game_state, float(episode_finished)))
        current_game_state = next_game_state
        current_episode_total_reward += immediate_reward

        if episode_finished:
            # Track episode return for moving average BEFORE reset/zeroing
            recent_episode_rewards.append(current_episode_total_reward)
            current_game_state, _ = environment.reset()
            current_episode_total_reward = 0.0

        # Train the neural network when we have enough experiences
        if len(experience_memory) >= training_batch_size:
            # Sample a batch of past experiences for learning
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = experience_memory.sample(training_batch_size)
            
            # Convert to tensors and move to device
            batch_states_tensor = torch.FloatTensor(batch_states).to(device)
            batch_actions_tensor = torch.LongTensor(batch_actions).to(device)
            batch_rewards_tensor = torch.FloatTensor(batch_rewards).to(device)
            batch_next_states_tensor = torch.FloatTensor(batch_next_states).to(device)
            batch_dones_tensor = torch.FloatTensor(batch_dones).to(device)

            # Compute target Q-values using the target network
            with torch.no_grad():
                target_network_predictions = target_agent_brain(batch_next_states_tensor)
                maximum_future_rewards = target_network_predictions.max(1)[0]
                target_q_values = batch_rewards_tensor + future_reward_discount_factor * (1.0 - batch_dones_tensor) * maximum_future_rewards

            # Compute current Q-values using the main network
            current_q_values = main_agent_brain(batch_states_tensor)
            selected_q_values = current_q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)

            # Compute prediction error and update the network
            prediction_error = prediction_error_function(selected_q_values, target_q_values)
            
            neural_network_optimizer.zero_grad()
            prediction_error.backward()
            neural_network_optimizer.step()

            # Cache scalar loss for logging
            last_training_loss_value = float(prediction_error.detach().cpu().item())

        # Update target network periodically to stabilize training
        if training_step % target_network_update_frequency == 0:
            target_agent_brain.load_state_dict(main_agent_brain.state_dict())

        # Periodic console log (kept cheap)
        if training_step % log_every_n_steps == 0:
            avg_reward = (sum(recent_episode_rewards) / len(recent_episode_rewards)) if len(recent_episode_rewards) > 0 else float('nan')
            print(
                f"step={training_step} eps={exploration_randomness:.3f} "
                f"replay={len(experience_memory)} batch={training_batch_size} "
                f"avg_ep_reward(last {len(recent_episode_rewards)}): {avg_reward:.3f} "
                f"loss={last_training_loss_value if last_training_loss_value is not None else float('nan'):.5f}"
            )

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(main_agent_brain.state_dict(), "models/dqn_pytorch_model.pth")
    print("Trained model saved to models/dqn_pytorch_model.pth")


if __name__ == "__main__":
    main()
