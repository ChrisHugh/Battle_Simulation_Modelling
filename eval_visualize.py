import time
import numpy as np

from tensorflow import keras

from battle_env_gym import BattleArenaEnv
from train_dqn import build_q_network


def main():
    env = BattleArenaEnv(max_steps=1200, role="archer", headless=False)
    obs, _ = env.reset()
    num_actions = env.action_space.n
    obs_dim = int(np.prod(obs.shape))

    model = build_q_network(obs_dim, num_actions)
    model = keras.models.load_model("models/dqn_tf_model.keras")

    for ep in range(3):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            env.render(mode="human")
            q_values = model.predict(obs[np.newaxis, :], verbose=0)[0]
            action = int(np.argmax(q_values))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.01)
        print(f"Episode {ep+1} total_reward={total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()


