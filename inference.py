from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from pong_rl.agent import AgentConfig, DeepQAgent
from pong_rl.environment import PongEnvironment
from pong_rl.processing import process_frame
from pong_rl.train_loop import append_frame, stack_initial_frames
from pong_rl.utils import set_random_seeds


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model_dir = Path(to_absolute_path(cfg.paths.model_dir))

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_dir}. Run train.py before inference."
        )

    seed = int(cfg.seed)
    enable_tf_determinism = True
    if "reproducibility" in cfg and "enable_tf_determinism" in cfg.reproducibility:
        enable_tf_determinism = bool(cfg.reproducibility.enable_tf_determinism)

    set_random_seeds(seed, enable_tf_determinism=enable_tf_determinism)

    agent_cfg = AgentConfig(**cfg.agent)
    agent = DeepQAgent(agent_cfg)
    agent.load_model(model_dir)

    env = PongEnvironment(render=True)

    try:
        _, initial_frame = env.reset()
        processed = process_frame(initial_frame)
        state = stack_initial_frames(processed)

        print("Starting inference. Close the window or press Ctrl+C to exit.")

        step = 0
        while True:
            action = agent.select_greedy_action(state)
            reward, raw_frame, smooth_score, done = env.step(action)
            processed = process_frame(raw_frame)
            state = append_frame(state, processed)

            if step % 25 == 0:
                print(
                    f"Step {step:05d} | Reward {reward:+.2f} | Smoothed Score {smooth_score:+.2f}"
                )

            step += 1

    except KeyboardInterrupt:
        print("Inference interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
