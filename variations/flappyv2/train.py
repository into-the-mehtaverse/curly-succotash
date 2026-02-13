"""
Train Flappy v2 curriculum with an LSTM policy.

This is isolated from existing runs:
- env code lives in variations/flappyv2/
- checkpoints/logs default to variations/flappyv2/experiments/

Usage (from repo root):
  uv run python -m variations.flappyv2.train
  uv run python -m variations.flappyv2.train --train.total-timesteps 150000000
  uv run python -m variations.flappyv2.train --train.load-checkpoint variations/flappyv2/experiments/<run_id>/model_XXXXXX.pt
"""

import argparse
import multiprocessing
import os
import sys

import torch
import pufferlib.models
import pufferlib.vector
from pufferlib import pufferl

from variations.flappyv2 import compute_difficulty, curriculum_env_creator


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "experiments")


def make_flappyv2_lstm_policy(env, hidden_size=128, lstm_hidden_size=128):
    base_policy = pufferlib.models.Default(env, hidden_size=hidden_size)
    return pufferlib.models.LSTMWrapper(
        env,
        base_policy,
        input_size=hidden_size,
        hidden_size=lstm_hidden_size,
    )


def _strip_arg(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        del sys.argv[i]
        if i < len(sys.argv) and not sys.argv[i].startswith("--"):
            del sys.argv[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train.total-timesteps", type=int, default=None, dest="train_total_timesteps")
    parser.add_argument("--train.load-checkpoint", type=str, default=None, dest="train_load_checkpoint")
    parser.add_argument("--train.learning-rate", type=float, default=None, dest="train_learning_rate")
    parser.add_argument("--train.output-dir", type=str, default=None, dest="train_output_dir")
    known, _ = parser.parse_known_args()

    _strip_arg("--train.total-timesteps")
    _strip_arg("--train.load-checkpoint")
    _strip_arg("--train.learning-rate")
    _strip_arg("--train.output-dir")

    args = pufferl.load_config("default")
    args["train"]["env"] = "flappyv2_curriculum"
    args["train"]["total_timesteps"] = known.train_total_timesteps if known.train_total_timesteps is not None else 150_000_000
    args["train"]["optimizer"] = "adam"
    args["train"]["learning_rate"] = known.train_learning_rate or 3e-4
    args["train"]["clip_coef"] = 0.2
    args["train"]["ent_coef"] = 0.01
    args["train"]["use_rnn"] = True
    args["train"]["data_dir"] = known.train_output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(args["train"]["data_dir"], exist_ok=True)
    print(f"[flappyv2] checkpoint dir: {args['train']['data_dir']}")

    if not torch.cuda.is_available():
        args["train"]["device"] = "cpu"

    vec_kwargs = dict(args["vec"])
    if vec_kwargs.get("num_workers") == "auto":
        vec_kwargs["num_workers"] = 2
    if vec_kwargs.get("num_envs") in (None, "auto") or vec_kwargs.get("num_envs", 0) < 128:
        vec_kwargs["num_envs"] = 128

    curriculum_difficulty_value = multiprocessing.Value("f", 0.0)
    vecenv = pufferlib.vector.make(
        curriculum_env_creator,
        env_kwargs={
            "num_envs": 1,
            "width": 400,
            "height": 600,
            "curriculum_difficulty_value": curriculum_difficulty_value,
        },
        **vec_kwargs,
    )

    policy = make_flappyv2_lstm_policy(vecenv.driver_env).to(args["train"]["device"])

    if known.train_load_checkpoint:
        state_dict = torch.load(known.train_load_checkpoint, map_location=args["train"]["device"])
        policy.load_state_dict(state_dict, strict=True)
        print(f"Loaded policy from {known.train_load_checkpoint} (fine-tuning)")

    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    total_ts = args["train"]["total_timesteps"]
    initial_lr = args["train"]["learning_rate"]
    final_lr = initial_lr / 6

    while trainer.epoch < trainer.total_epochs:
        frac = min(1.0, trainer.global_step / max(1, total_ts))
        curriculum_difficulty_value.value = compute_difficulty(trainer.global_step, total_ts)
        lr = initial_lr + (final_lr - initial_lr) * frac
        if hasattr(trainer, "optimizer"):
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = lr
        trainer.evaluate()
        trainer.train()
        trainer.print_dashboard()

    trainer.close()
    print(f"Training finished. Check {args['train']['data_dir']}/ for checkpoints.")


if __name__ == "__main__":
    main()
