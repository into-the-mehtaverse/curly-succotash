# Flappy RL

Flappy Bird reinforcement learning environment built with [PufferLib](https://puffer.ai), a very fast RL training library.

The best and final setup in this repo is `variations/flappyv3`.

On my personal machine (MacBook), this setup converges very quickly (typically under 5-6 minutes for a strong policy). The eval environment is capped at 65 pipes for practical runtime.

If you want the full write-up of the process, read: [`docs/blog2.md`](docs/blog2.md).

## Quick Start (v3 only)

```bash
uv sync
cd variations/flappyv3 && make clean && make PYTHON=../../.venv/bin/python && cd ../..
```

### Train (best/final version)

```bash
uv run python -m variations.flappyv3.train
```

### Headless eval (50 games)

```bash
uv run python -m variations.flappyv3.run_eval --episodes 50 --no-render
```

### Render eval with latest checkpoint

```bash
uv run python -m variations.flappyv3.run_eval
```

## Notes To Self

- Older curriculum checkpoints (pre-v3):
  - `experiments/177086914642/model_009765.pt` (80M-step run)
  - `experiments/177087020156/model_018310.pt` (150M-step run)
- v3 checkpoints are stored under:
  - `variations/flappyv3/experiments/<run_id>/model_XXXXXX.pt`
