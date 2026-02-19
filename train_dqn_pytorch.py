from math import e
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

# Multiprocessing imports
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock
import threading
import time


class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
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


class SharedReplayBuffer:
    """
    Thread-safe shared experience replay buffer for parallel training.
    Multiple workers can add experiences while a main process samples for training.
    """
    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0
        self.lock = Lock() 
        
    def add(self, transition):
        """
        Add a new experience to the buffer in a thread-safe manner.
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.idx] = transition
            self.idx = (self.idx + 1) % self.capacity
        
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer in a thread-safe manner.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of numpy arrays (states, actions, rewards, next_states, dones)
            or None if not enough experiences
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            
            # Sample random indices
            idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            batch = [self.buffer[i] for i in idxs]
            
            # Convert to numpy arrays and return as tuple
            return map(np.array, zip(*batch))
        
    def __len__(self):
        """
        Get the current number of experiences in the buffer.
        
        Returns:
            Number of experiences stored
        """
        with self.lock:
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


def worker_training_function(worker_id, experience_queue, model_params_queue, 
                            episode_reward_queue, device, num_steps_per_worker):
    """
    Worker function that runs a separate environment and collects experiences.
    Each worker runs independently and adds experiences to the shared buffer.
    
    Args:
        worker_id: Unique identifier for this worker
        shared_buffer: Shared experience replay buffer
        model_params_queue: Queue to receive updated model parameters
        exploration_rate: Current exploration rate
        device: PyTorch device (CPU for workers, GPU for main)
        num_steps_per_worker: Number of steps each worker should run
    """
    environment = BattleArenaEnv(max_steps=1000, role="archer", headless=True)
    num_actions = environment.action_space.n  # Fix: .n gets the number of actions
    observation, _ = environment.reset()
    observation_size = int(np.prod(observation.shape))
    exploration_rate = 0.995
    
    # Create local network (CPU-based, lighter than GPU version)
    local_network = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    
    # Wait for initial model parameters from main process
    print(f"Worker {worker_id}: Waiting for initial model parameters...")
    initial_params = model_params_queue.get()  # This blocks until main sends params
    local_network.load_state_dict(initial_params)
    local_network.eval()  # Set to evaluation mode (no gradients needed)
    print(f"Worker {worker_id}: Received initial model parameters!")
    
    # Worker training variables
    current_state = observation
    episode_reward = 0.0
    steps_collected = 0
    
    print(f"Worker {worker_id}: Starting to collect {num_steps_per_worker} steps")
    print(f"Worker {worker_id}: Using device: {device}")
    print(f"Worker {worker_id}: Exploration rate: {exploration_rate}")
    
    while steps_collected < num_steps_per_worker:
        # Check for updated model parameters (non-blocking)
        if not model_params_queue.empty():
            updated_params = model_params_queue.get()
            local_network.load_state_dict(updated_params)
            local_network.eval()
            print(f"Worker {worker_id}: Updated model parameters at step {steps_collected}")
        
        # Choose action (explore or exploit)
        if np.random.rand() < exploration_rate:
            chosen_action = environment.action_space.sample()
        else:
            game_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            with torch.no_grad():
                predicted_action_values = local_network(game_state_tensor).cpu().numpy()[0]
            chosen_action = int(np.argmax(predicted_action_values))
        
        # Step environment
        next_game_state, immediate_reward, episode_terminated, episode_truncated, _ = environment.step(chosen_action)
        episode_finished = episode_terminated or episode_truncated
        episode_reward += immediate_reward
        
        # Add experience to shared buffer (store current state, not next state)
        experience_queue.put((current_state, chosen_action, immediate_reward, next_game_state, float(episode_finished)))
        
        # Update current state
        current_state = next_game_state
        
        # Handle episode termination
        if episode_finished:
            # Send episode reward to main process
            episode_reward_queue.put((worker_id, episode_reward))
            current_state, _ = environment.reset()
            episode_reward = 0.0
        
        # Update step counter
        exploration_rate = max(0.05,exploration_rate*0.995)
        steps_collected += 1
    
    print(f"Worker {worker_id}: Finished collecting {steps_collected} steps")


def parallel_training_main(num_workers=6):
    """
    Main function for parallel training with multiple workers.
    Coordinates multiple workers and handles centralized learning.
    
    Args:
        num_workers: Number of worker processes to start
    """
    print("=" * 60)
    print("PARALLEL TRAINING SETUP")
    print("=" * 60)
    
    # Device set-up, CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main process using device: {device}")
    print(f"Number of workers: {num_workers}")
    
    # Initialize environment and get information
    print("\nGetting environment information...")
    temp_env = BattleArenaEnv(max_steps=1000, role="archer", headless=True)
    num_actions = temp_env.action_space.n
    observation, _ = temp_env.reset()
    observation_size = int(np.prod(observation.shape))
    temp_env.close()  # Close temporary environment
    print(f"Observation size: {observation_size}")
    print(f"Number of actions: {num_actions}")
    
    # Shared experience buffer
    print("\nCreating shared experience buffer...")
    shared_buffer = SharedReplayBuffer(capacity=50000)
    print(f"Shared buffer capacity: {shared_buffer.capacity}")
    
    # Main & target neural network on GPU
    print("\nCreating neural networks on GPU...")
    main_agent_network = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    target_agent_network = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    target_agent_network.load_state_dict(main_agent_network.state_dict())
    print("Main and target networks created and synchronized on GPU ✓")
    
    # Model optimizer and loss functions
    print("\nSetting up training components...")
    optimizer = optim.Adam(main_agent_network.parameters(), lr=1e-3)
    loss_function = nn.SmoothL1Loss()
    print("Optimizer and loss function ready ✓")

    # Training hyperparameters
    future_reward_discount_factor = 0.999
    training_batch_size = 512  # Reduced batch size for faster training start
    model_update_frequency = 2048  # Send updated model to workers every 1000 steps

    # Exploration hyperparameters
    initial_exploration_rate = 1.0

    # Env & Training Steps
    total_environment_steps = 3_000_000
    
    # Communication queues for worker models
    print("\nCreating communication queues...")
    model_queues = [Queue() for _ in range(num_workers)]
    episode_reward_queue = Queue()  # Queue for episode rewards from workers
    experience_queue = Queue()
    
    # Start worker processes
    print(f"\nStarting {num_workers} worker processes on CPU...")
    workers = []
    for worker_id in range(num_workers):
        worker = Process(target=worker_training_function,
                        args=(worker_id, experience_queue, model_queues[worker_id],
                             episode_reward_queue, "cpu", (total_environment_steps//num_workers)))
        workers.append(worker)
        worker.start()


    # Send initial model parameters to workers
    print("\nSending initial model parameters to workers...")
    initial_params = main_agent_network.state_dict()
    # Convert CUDA tensors to CPU tensors for workers
    cpu_params = {k: v.cpu() for k, v in initial_params.items()}
    for queue in model_queues:
        queue.put(cpu_params)
    print(f"Sent initial parameters to {num_workers} workers ✓")
    
    # Main training loop
    print("\nStarting main training loop...")

    # Progress tracking
    training_step = 0
    max_training_step = 2000
    recent_episode_rewards = deque(maxlen=20)
    last_training_loss_value = float('nan')
    
    print(f"Training for {total_environment_steps} environment steps...")
    print(f"Batch size: {training_batch_size}")
    print(f"Model updates every {model_update_frequency} steps")
    print("Press Ctrl+C to stop training early")
    
    # Main training loop
    try:
        while training_step < max_training_step:
            # Collect episode rewards from workers
            while not episode_reward_queue.empty():
                worker_id, episode_reward = episode_reward_queue.get()
                recent_episode_rewards.append(episode_reward)
            
            while not experience_queue.empty():
                experience = experience_queue.get()
                shared_buffer.add(experience)
        
            # Sample batch from shared buffer
            if len(shared_buffer) >= training_batch_size:
                batch_data = shared_buffer.sample(training_batch_size)
                if batch_data is None:
                    continue
                if batch_data is not None:
                    print(f"Training at step {training_step} with buffer size {len(shared_buffer)}")
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch_data
                    
                    # Convert to tensors and move to device
                    batch_states_tensor = torch.FloatTensor(batch_states).to(device)
                    batch_actions_tensor = torch.LongTensor(batch_actions).to(device)
                    batch_rewards_tensor = torch.FloatTensor(batch_rewards).to(device)
                    batch_next_states_tensor = torch.FloatTensor(batch_next_states).to(device)
                    batch_dones_tensor = torch.FloatTensor(batch_dones).to(device)
                    
                    # Train neural network
                    with torch.no_grad():
                        target_network_predictions = target_agent_network(batch_next_states_tensor)
                        maximum_future_rewards = target_network_predictions.max(1)[0]
                        target_q_values = batch_rewards_tensor + future_reward_discount_factor * (1.0 - batch_dones_tensor) * maximum_future_rewards
                    
                    # Compute current Q-values using main network
                    current_q_values = main_agent_network(batch_states_tensor)
                    selected_q_values = current_q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)
                    
                    # Compute loss and update network
                    loss = loss_function(selected_q_values, target_q_values)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Cache loss for logging
                    last_training_loss_value = float(loss.detach().cpu().item())
                    training_step += 1
            
                    # Update target network periodically
                    if training_step % 4 == 0:
                        target_agent_network.load_state_dict(main_agent_network.state_dict())
                    
                    # Send updated model to workers periodically
                    if training_step % 2 == 0:
                        updated_params = main_agent_network.state_dict()
                        # Convert CUDA tensors to CPU tensors for workers
                        cpu_params = {k: v.cpu() for k, v in updated_params.items()}
                        for queue in model_queues:
                            queue.put(cpu_params)
                        print(f"Step {training_step}: Sent updated model to all workers")
        
                    # Log progress
                    if training_step % 2 == 0:
                        avg_reward = (sum(recent_episode_rewards) / len(recent_episode_rewards)) if len(recent_episode_rewards) > 0 else float('nan')
                        print(
                            f"step={training_step}"
                            f"replay={len(shared_buffer)} batch={training_batch_size} "
                            f"avg_ep_reward(last {len(recent_episode_rewards)}): {avg_reward:.3f} "
                            f"loss={last_training_loss_value if last_training_loss_value is not None else float('nan'):.5f}"
                        )
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C)")
    
    # Terminate all workers
    print("\nTerminating workers...")
    for worker in workers:
        worker.terminate()  # Force terminate workers
        worker.join()       # Wait for termination to complete
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(main_agent_network.state_dict(), "models/dqn_pytorch_parallel_model.pth")
    print("Trained parallel model saved to models/dqn_pytorch_parallel_model.pth")
    
    print("\n" + "=" * 60)
    print("PARALLEL TRAINING COMPLETE")
    print("=" * 60)
    
    # Force exit to ensure clean termination
    import sys
    sys.exit(0)


def main():
    """
    Main entry point - chooses between single-threaded and parallel training.
    """
    # Training mode selection
    use_parallel_training = True  # Set to True for parallel training
    num_workers = 6  # Number of parallel workers (adjust based on CPU cores)
    
    if use_parallel_training:
        print(f"Starting parallel training with {num_workers} workers...")
        parallel_training_main(num_workers)
        return
    
    print("Starting single-threaded training...")
    single_threaded_training()


def single_threaded_training():
    """
    Original single-threaded training implementation.
    This is your existing training code.
    """
    
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
    prediction_error_function = nn.SmoothL1Loss()

    # Experience replay buffer for storing past game experiences
    experience_memory = ReplayBuffer(capacity=200000)
    
    # Training hyperparameters
    future_reward_discount_factor = 0.999
    training_batch_size = 512
    exploration_randomness = 1.0
    minimum_exploration_randomness = 0.05
    exploration_decay_rate = 0.999995
    target_network_update_frequency = 512
    total_training_steps = 10000000

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
    mp.set_start_method('spawn', force=True)  # Required for Windows
    main()
