import os
import numpy as np
import torch
import torch.nn as nn

from battle_env_gym import BattleArenaEnv


class BattleAgentNeuralNetwork(nn.Module):
    def __init__(self, observation_vector_size: int, number_of_possible_actions: int):
        super(BattleAgentNeuralNetwork, self).__init__()
        self.observation_to_features_layer = torch.nn.Linear(observation_vector_size, 128)
        self.feature_processing_layer = torch.nn.Linear(128, 128)
        self.action_value_prediction_layer = torch.nn.Linear(128, number_of_possible_actions)

    def forward(self, observation_tensor: torch.Tensor) -> torch.Tensor:
        hidden_features_1 = torch.relu(self.observation_to_features_layer(observation_tensor))
        hidden_features_2 = torch.relu(self.feature_processing_layer(hidden_features_1))
        action_values = self.action_value_prediction_layer(hidden_features_2)
        return action_values


def get_observation_names():
    # Must match battle_env_gym._get_observation() ordering
    return [
        "agent_hp",
        "agent_x_norm",
        "agent_y_norm",
        "boss_hp",
        "rel_x_norm",
        "rel_y_norm",
        "distance_norm",
        "active_ability_id_norm",
        "ability_ticks_norm",
        "step_progress",
        "agent_speed_norm",
    ]


def get_action_names():
    # Matches BattleArenaEnv._action_to_unit_vector mapping
    return [
        "stay",
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
        "attack",
    ]


def format_observation_dict(obs: np.ndarray) -> dict:
    names = get_observation_names()
    return {names[i]: float(obs[i]) for i in range(len(names))}


def decode_ability_id_norm(id_value, tick_value) -> str:
    # In env: name_to_id = {"Frontal Cone": 0, "Tank Buster": 1, "Fireball": 2}; id normalized by /2
    if id_value <= 0.0 and tick_value == 0:
        return "None"
    elif id_value == 0.0 and tick_value >0:
        return "Frontal cone"
    elif 0.0 < id_value <= (1.0/2.0):
        return "Tank Buster (~1)"
    return "Fireball (~2)"


def maybe_plot_q_values(q_values: np.ndarray, action_names: list, title: str = "Q-values"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    indices = np.arange(len(q_values))
    plt.figure(figsize=(8, 3))
    plt.bar(indices, q_values, color="#4C78A8")
    plt.xticks(indices, action_names, rotation=30, ha="right")
    plt.ylabel("Q-value")
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment
    env = BattleArenaEnv(max_steps=300, role="archer", headless=True)
    action_names = get_action_names()

    # Observation shape
    observation, _ = env.reset()
    observation_size = int(np.prod(observation.shape))
    num_actions = env.action_space.n

    # Load model
    model_path = os.path.join("models", "dqn_pytorch_model.pth")
    model = BattleAgentNeuralNetwork(observation_size, num_actions).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model: {model_path}")
    else:
        print("Warning: model file not found; using randomly initialized network.")

    # One episode inspection
    state = observation
    total_reward = 0.0

    print("\n=== Inspection: printing what the model sees and decides each step ===")
    step = 0
    while True:
        step += 1

        # Pretty observation
        obs_dict = format_observation_dict(state)
        # Human-friendly ability decode (keep separate so we don't format as float)
        ability_label = decode_ability_id_norm(obs_dict["active_ability_id_norm"],obs_dict['ability_ticks_norm'])

        # Model prediction
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_tensor = model(state_tensor)
        q_values = q_tensor.squeeze(0).detach().cpu().numpy()
        greedy_action = int(np.argmax(q_values))

        # Print
        print(f"\nStep {step}")
        print("Observation (named):")
        for k, v in obs_dict.items():
            try:
                print(f"  - {k}: {float(v):.4f}")
            except Exception:
                print(f"  - {k}: {v}")
        print(f"  - ability_decoded: {ability_label}")
        print("Q-values (action -> value):")
        for i, name in enumerate(action_names):
            print(f"  - {i:>1} {name:<10}: {q_values[i]:.4f}")
        print(f"Chosen (greedy): {greedy_action} ({action_names[greedy_action]})")

        #maybe_plot_q_values(q_values, action_names, title=f"Q-values at step {step}")

        # Step environment with chosen action
        next_state, reward, terminated, truncated, _ = env.step(greedy_action)
        total_reward += reward
        print(f"Reward this step: {reward:.4f}   Total: {total_reward:.4f}")

        state = next_state
        if terminated or truncated or step >= 30:
            print("\nEpisode finished.")
            break


if __name__ == "__main__":
    main()


