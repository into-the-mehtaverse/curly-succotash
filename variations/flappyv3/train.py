"""
Train Flappy v3 with Target-style policy/hyperparameters.

Key differences from v2:
- No curriculum ramp (fixed difficulty throughout training)
- No custom/manual LR schedule (uses PuffeRL defaults like Target)
- LSTM policy (Default + LSTMWrapper), matching Target policy family
- Outputs are isolated under variations/flappyv3/experiments/

Usage (from repo root):
  uv run python -m variations.flappyv3.train
  uv run python -m variations.flappyv3.train --train.total-timesteps 100000000
  uv run python -m variations.flappyv3.train --train.load-checkpoint variations/flappyv3/experiments/<run_id>/model_XXXXXX.pt
"""

import argparse
import multiprocessing
import os
import sys

import torch
import pufferlib.models
import pufferlib.vector
from pufferlib import pufferl

from variations.flappyv3 import curriculum_env_creator


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "experiments")


def make_flappyv3_lstm_policy(env, hidden_size=128, lstm_hidden_size=128):
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
    parser.add_argument("--env.fixed-difficulty", type=float, default=1.0, dest="env_fixed_difficulty")
    known, _ = parser.parse_known_args()

    _strip_arg("--train.total-timesteps")
    _strip_arg("--train.load-checkpoint")
    _strip_arg("--train.learning-rate")
    _strip_arg("--train.output-dir")
    _strip_arg("--env.fixed-difficulty")

    args = pufferl.load_config("default")
    args["train"]["env"] = "flappyv3_targetlike"
    args["train"]["total_timesteps"] = known.train_total_timesteps if known.train_total_timesteps is not None else 100_000_000
    args["train"]["optimizer"] = "muon"
    args["train"]["learning_rate"] = known.train_learning_rate or 0.015
    args["train"]["gamma"] = 0.99
    args["train"]["minibatch_size"] = 32768
    args["train"]["ent_coef"] = 0.02
    args["train"]["anneal_lr"] = True
    args["train"]["use_rnn"] = True
    args["train"]["data_dir"] = known.train_output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(args["train"]["data_dir"], exist_ok=True)
    print(f"[flappyv3] checkpoint dir: {args['train']['data_dir']}")

    if not torch.cuda.is_available():
        args["train"]["device"] = "cpu"

    vec_kwargs = dict(args["vec"])
    if vec_kwargs.get("num_workers") == "auto":
        vec_kwargs["num_workers"] = 2
    if vec_kwargs.get("num_envs") in (None, "auto") or vec_kwargs.get("num_envs", 0) < 128:
        vec_kwargs["num_envs"] = 128

    # No curriculum in v3: keep difficulty fixed for the whole run.
    difficulty_value = multiprocessing.Value("f", float(known.env_fixed_difficulty))
    vecenv = pufferlib.vector.make(
        curriculum_env_creator,
        env_kwargs={
            "num_envs": 1,
            "width": 400,
            "height": 600,
            "curriculum_difficulty_value": difficulty_value,
        },
        **vec_kwargs,
    )

    policy = make_flappyv3_lstm_policy(vecenv.driver_env).to(args["train"]["device"])

    if known.train_load_checkpoint:
        state_dict = torch.load(known.train_load_checkpoint, map_location=args["train"]["device"])
        policy.load_state_dict(state_dict, strict=True)
        print(f"Loaded policy from {known.train_load_checkpoint} (fine-tuning)")

    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        trainer.train()
        trainer.print_dashboard()

    trainer.close()
    print(f"Training finished. Check {args['train']['data_dir']}/ for checkpoints.")


if __name__ == "__main__":
    main()
