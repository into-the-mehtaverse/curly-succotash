"""
Train the PufferLib Target (ocean) environment as a baseline to compare convergence.

All outputs (checkpoints, trainer state) go under puffer_target_baseline/experiments/
so they don't mix with your Flappy runs.

Usage (from repo root):
  uv run python puffer_target_baseline/train_target.py
  uv run python puffer_target_baseline/train_target.py --train.total-timesteps 20_000_000

Override any config with --section.key value (e.g. --train.learning-rate 0.01).
"""

import os
import sys

# Run from repo root so data_dir is correct
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != REPO_ROOT:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "puffer_target_baseline", "experiments")


def main():
    import argparse
    import pufferlib
    from pufferlib import pufferl

    # Parse a few args before load_config so we can strip them (pufferl rejects unknown)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train.total-timesteps", type=int, default=None, dest="total_timesteps")
    known, _ = parser.parse_known_args()
    for i in list(reversed(range(len(sys.argv)))):
        if sys.argv[i].startswith("--train.total-timesteps"):
            del sys.argv[i]
            if i < len(sys.argv) and not sys.argv[i].startswith("--"):
                del sys.argv[i]
            break

    # Load PufferLib's config for the Target env (puffer_target)
    args = pufferl.load_config("puffer_target")

    # Send all checkpoints and state into our folder
    args["train"]["data_dir"] = EXPERIMENTS_DIR
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    if known.total_timesteps is not None:
        args["train"]["total_timesteps"] = known.total_timesteps

    # CPU-friendly defaults when no CUDA (e.g. MacBook)
    if not __import__("torch").cuda.is_available():
        args["train"]["device"] = "cpu"
        # Slightly smaller batch to avoid OOM on laptop
        if args["vec"].get("num_envs", 0) > 512:
            args["vec"]["num_envs"] = 512

    pufferl.train(env_name="puffer_target", args=args)
    print("Done. Checkpoints and logs in:", EXPERIMENTS_DIR)


if __name__ == "__main__":
    main()
