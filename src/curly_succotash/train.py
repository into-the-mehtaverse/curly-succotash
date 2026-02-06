"""
Train an agent on FlappyGridEnv with PufferLib's PuffeRL.

Usage:
  uv run python -m curly_succotash.train
  uv run python -m curly_succotash.train --train.device cuda --train.total_timesteps 1000000
"""

import torch
import pufferlib
import pufferlib.vector
from pufferlib import pufferl

from curly_succotash.env import flappy_grid_env_creator, FlappyGridEnv


class FlappyGridPolicy(torch.nn.Module):
    """Simple MLP policy for FlappyGrid (small obs, discrete actions)."""

    def __init__(self, env):
        super().__init__()
        obs_size = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.n
        hidden = 64
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, hidden)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(hidden, hidden)),
            torch.nn.ReLU(),
        )
        self.action_head = torch.nn.Linear(hidden, n_actions)
        self.value_head = torch.nn.Linear(hidden, 1)

    def forward_eval(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


def main():
    # Load default PufferLib config (train + vec sections)
    args = pufferl.load_config("default")
    args["train"]["env"] = "flappy_grid"
    # Long enough to see policy learn (position + small survival reward); override with --train.total_timesteps if needed
    args["train"]["total_timesteps"] = 2_000_000
    # Use Adam so we don't require the optional muon/heavyball dependency
    args["train"]["optimizer"] = "adam"
    # Higher LR + looser clip for this trivial task (learn the "don't hit wall" rule fast)
    args["train"]["learning_rate"] = 0.01
    args["train"]["clip_coef"] = 0.5  # allow bigger policy updates (default 0.2)
    # Prefer CPU so it runs without GPU; use --train.device cuda for GPU
    if not torch.cuda.is_available():
        args["train"]["device"] = "cpu"

    # Vectorized env: need batch_size (= num_envs * bptt_horizon) >= minibatch_size (default 8192)
    vec_kwargs = dict(args["vec"])
    if vec_kwargs.get("num_workers") == "auto":
        vec_kwargs["num_workers"] = 2
    if vec_kwargs.get("num_envs") in (None, "auto") or vec_kwargs.get("num_envs", 0) < 128:
        vec_kwargs["num_envs"] = 128  # 128 * 64 (bptt_horizon) = 8192 >= minibatch_size
    vecenv = pufferlib.vector.make(
        flappy_grid_env_creator,
        **vec_kwargs,
    )
    policy = FlappyGridPolicy(vecenv.driver_env).to(args["train"]["device"])

    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        trainer.train()
        trainer.print_dashboard()

    trainer.close()
    print("Training finished. Check experiments/ for checkpoints.")


if __name__ == "__main__":
    main()
