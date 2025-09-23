import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from battle_env_gym import BattleArenaEnv


class BattleAgentNeuralNetwork(nn.Module):
    """
    Deep Q-Network for battle simulation agent.
    
    This network learns to predict the value of each possible action
    given the current game state (observation).
    
    Architecture:
    - Input: 12-dimensional observation vector (agent health, position, boss info, etc.)
    - Hidden Layer 1: 128 neurons with ReLU activation
    - Hidden Layer 2: 128 neurons with ReLU activation  
    - Output: 9 Q-values (one for each possible action: no-op + 8 movement directions)
    """
    
    def __init__(self, observation_vector_size: int, number_of_possible_actions: int):
        super(BattleAgentNeuralNetwork, self).__init__()
        
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
        # Transform raw observations into meaningful features
        hidden_features_1 = torch.relu(self.observation_to_features_layer(observation_tensor))
        
        # Process features to understand complex game patterns
        hidden_features_2 = torch.relu(self.feature_processing_layer(hidden_features_1))
        
        # Predict how good each action would be in this state
        action_values = self.action_value_prediction_layer(hidden_features_2)
        
        return action_values


def load_model(model_path: str, observation_size: int, num_actions: int, device: str = "cpu"):
    """Load a trained PyTorch model"""
    model = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Evaluation-only exploration (does not affect training)
    eval_epsilon = 0.05

    # Create environment
    environment = BattleArenaEnv(max_steps=600, role="archer", headless=False)
    num_actions = environment.action_space.n
    observation, _ = environment.reset()
    observation_size = int(np.prod(observation.shape))

    # Load trained model
    model_path = "models/dqn_pytorch_model.pth"
    try:
        model = load_model(model_path, observation_size, num_actions, device)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train a model first.")
        return

    # Run evaluation
    state = observation
    total_reward = 0
    step = 0
    
    while True:
        # Get action from model with optional evaluation exploration
        if np.random.rand() < eval_epsilon:
            action = environment.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(torch.argmax(model(state_tensor), dim=1).item())

        # Take step
        next_state, reward, terminated, truncated, _ = environment.step(action)
        state = next_state
        done = terminated or truncated
        
        total_reward += reward
        step += 1

        # Render
        environment.overlay_text = (
            f"Step: {step}  Reward: {reward:.2f}  Total: {total_reward:.2f}  "
            f"Action: {action}  Device: {device}"
        )
        environment.render(mode="human")

        if done:
            print(f"Episode finished! Total reward: {total_reward:.2f}")
            state, _ = environment.reset()
            total_reward = 0
            step = 0

        # Break on user input (you can modify this)
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return


if __name__ == "__main__":
    main()
