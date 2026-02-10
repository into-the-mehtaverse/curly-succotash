"""
Eval Flappy with a trained policy (saved checkpoint). Renders the game.
Run from repo root so C code finds resources/flappy/:

  uv run python -m curly_succotash.run_eval_flappy
  uv run python -m curly_succotash.run_eval_flappy --model experiments/<run_id>/model_000610.pt

Press ESC in the game window to exit.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import pufferlib.vector
import pufferlib.pytorch

from curly_succotash.flappy import flappy_env_creator
from curly_succotash.train import FlappyGridPolicy

FPS = 60


def find_latest_checkpoint():
    pattern = "experiments/*/model_*.pt"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def main():
    parser = argparse.ArgumentParser(description="Eval Flappy with a trained policy")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to checkpoint .pt (default: latest in experiments/)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = args.model or find_latest_checkpoint()
    if not model_path or not os.path.isfile(model_path):
        print("No checkpoint found. Train first or pass --model path/to/model_XXXXXX.pt")
        return

    vecenv = pufferlib.vector.make(
        flappy_env_creator,
        env_kwargs={"num_envs": 1, "width": 400, "height": 600},
        backend=pufferlib.vector.Serial,
        num_envs=1,
        seed=args.seed,
    )
    driver = vecenv.driver_env
    policy = FlappyGridPolicy(driver).to(args.device)
    state_dict = torch.load(model_path, map_location=args.device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    obs, info = vecenv.reset(seed=args.seed)
    with torch.no_grad():
        while True:
            driver.render()
            ob = torch.as_tensor(obs).to(args.device)
            logits, _ = policy.forward_eval(ob)
            action = logits.argmax(dim=-1).cpu().numpy().reshape(vecenv.action_space.shape)
            obs, rewards, terms, truncs, info = vecenv.step(action)
            time.sleep(1 / FPS)


if __name__ == "__main__":
    main()
