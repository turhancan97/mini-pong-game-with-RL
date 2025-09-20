from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from pong_rl.agent import AgentConfig, DeepQAgent
from pong_rl.environment import PongEnvironment
from pong_rl.processing import process_frame
from pong_rl.train_loop import TrainingHistory, append_frame, stack_initial_frames
from pong_rl.utils import set_random_seeds


def _prepare_agent(cfg: DictConfig, checkpoint_dir: Path) -> Tuple[DeepQAgent, int]:
    agent_cfg = AgentConfig(**cfg.agent)
    agent = DeepQAgent(agent_cfg)
    start_step = 0

    if cfg.training.resume:
        restored = agent.load_checkpoint(checkpoint_dir)
        if restored:
            start_step = agent.steps
            print(f"Resuming training from checkpoint with step {start_step}.")

    return agent, start_step


def _save_scores(plot_dir: Path, history: TrainingHistory, plot_scores: bool) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    scores_path = plot_dir / "scores.pkl"

    with open(scores_path, "wb") as fh:
        pickle.dump(
            {
                "steps": history.steps,
                "scores": history.scores,
                "losses": history.losses,
            },
            fh,
        )
    print(f"Saved score history to {scores_path}.")

    if not plot_scores:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping score plot.")
        return

    if not history.steps:
        print("No scores recorded yet; skipping plot generation.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(history.steps, history.scores)
    plt.title("Training Score (Smoothed)")
    plt.xlabel("Training Step")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plot_path = plot_dir / "scores.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved score plot to {plot_path}.")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    checkpoint_dir = Path(to_absolute_path(cfg.paths.checkpoint_dir))
    model_dir = Path(to_absolute_path(cfg.paths.model_dir))
    plot_dir = Path(to_absolute_path(cfg.paths.plot_dir))

    seed = int(cfg.seed)
    enable_tf_determinism = True
    if "reproducibility" in cfg and "enable_tf_determinism" in cfg.reproducibility:
        enable_tf_determinism = bool(cfg.reproducibility.enable_tf_determinism)

    set_random_seeds(seed, enable_tf_determinism=enable_tf_determinism)

    agent, start_step = _prepare_agent(cfg, checkpoint_dir)

    total_steps = cfg.training.total_steps
    if start_step >= total_steps:
        print(
            "Checkpoint already reached configured total steps; nothing to train."
        )
        return

    env = PongEnvironment(render=cfg.training.render)
    history = TrainingHistory()

    try:
        _, initial_frame = env.reset()
        processed_frame = process_frame(initial_frame)
        state = stack_initial_frames(processed_frame)

        for step in range(start_step, total_steps):
            action = agent.select_action(state)
            reward, raw_frame, smooth_score, done = env.step(action)
            processed = process_frame(raw_frame)
            next_state = append_frame(state, processed)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state

            if step % cfg.training.log_interval == 0:
                print(
                    f"Step {step:06d} | Reward {reward:+.2f} | "
                    f"Smoothed Score {smooth_score:+.2f} | Loss {loss:.6f}"
                )
                history.record(step, smooth_score, loss)

            if (step + 1) % cfg.training.checkpoint_interval == 0:
                agent.save_checkpoint(checkpoint_dir)
                _save_scores(plot_dir, history, cfg.logging.plot_scores)

        print("Training complete; saving final artifacts.")
        agent.save_model(model_dir)
        agent.save_checkpoint(checkpoint_dir)

    finally:
        env.close()

    _save_scores(plot_dir, history, cfg.logging.plot_scores)


if __name__ == "__main__":
    main()
