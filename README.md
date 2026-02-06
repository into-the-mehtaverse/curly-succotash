# Curly-succotash

First RL env for [PufferLib](https://puffer.ai): high-throughput reinforcement learning (millions of steps/second).

## Setup

**Requires:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

From the repo root:

```bash
uv sync
```

*(If `uv sync` fails in the IDE due to cache permissions, run it in your system terminal.)*

## What’s in this repo

- **`QUICKSTART.md`** — **Start here.** Short orientation to Gymnasium and PufferLib and what actually matters.
- **`RL_BASICS.md`** — New to RL? Read this for the big picture (agent, env, reward, training) and how this repo fits in.
- **`src/curly_succotash/env.py`** — Envs:
  1. **Flappy Grid:** `FlappyGridEnv` — 2-row grid, up/down actions, wall-on-roof/floor obs, -1 reward for hitting ceiling (or floor). Use `flappy_grid_env_creator` with PufferLib vector/train.
  2. **Gymnasium + wrapper:** `SampleGymnasiumEnv` + `make_gymnasium_env()` (easiest).
  3. **Native PufferEnv:** `SamplePufferEnv` (in-place buffer updates for vectorization).
- **`src/curly_succotash/train.py`** — Train an agent on Flappy Grid with PuffeRL (PPO-style).
- **`src/curly_succotash/sweep.py`** — Sweep over learning rate and clip_coef; reports final entropy (deterministic when &lt; 0.01).
- **`ENV_INSTRUCTIONS.md`** — Step-by-step instructions (from PufferLib docs) for writing envs and using the API.

## Run the env demo (mock envs)

```bash
uv run python -m curly_succotash
```

## Flappy Grid: train an agent

```bash
uv run python -m curly_succotash.train
```

- Uses **FlappyGridEnv** (2-row grid, up/down, wall obs, -1 for hitting ceiling/floor) and PufferLib’s **PuffeRL** (PPO-style).
- Default: 500k steps, CPU (use `--train.device cuda` for GPU). Override e.g. `--train.total_timesteps 1000000`.
- Checkpoints go to **`experiments/`**.

## Sweep: when does the policy become deterministic?

```bash
uv run python -m curly_succotash.sweep
```

Runs training for each combination of learning rate and clip coefficient (see `sweep.py` for the grid), records final entropy per run, and prints a table. Entropy &lt; 0.01 is treated as deterministic. Edit `LEARNING_RATES` and `CLIP_COEFS` in `sweep.py` to change the grid; `SWEEP_TIMESTEPS` controls steps per run (default 500k).

*(If you see a Gym deprecation message, it comes from PufferLib’s dependencies; this project uses Gymnasium only. A RuntimeWarning about `curly_succotash.env` is avoided by using `python -m curly_succotash` instead of `python -m curly_succotash.env`.)*

## Docs and API

- Full PufferLib docs: **https://puffer.ai/docs.html**
- Env-writing tutorial and API summary: **`ENV_INSTRUCTIONS.md`** in this repo.
