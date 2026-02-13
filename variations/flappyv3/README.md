# Flappy v3 target-like baseline (C + raylib)

Copy of `variations/flappyv2` with a dedicated v3 training path to mirror Target:
- LSTM policy (Default + LSTMWrapper)
- Target-like train hyperparameters
- No curriculum ramp (fixed difficulty throughout run)
- No custom/manual LR decay (uses PuffeRL default scheduler behavior)

All v3 checkpoints are isolated under `variations/flappyv3/experiments/`.

## Build

1. **Install raylib** (required for rendering):
   - macOS: `brew install raylib`
   - Linux: install `libraylib-dev` (or build from [raylib](https://www.raylib.com/)).

2. **Build the C extension** from repo root (use the venv Python so the .so loads):
   ```bash
  cd variations/flappyv3
   make PYTHON=../../.venv/bin/python
   ```
   Or from `src/flappy_rl/flappy` for the main env:
   ```bash
   cd src/flappy_rl/flappy
   make PYTHON=../../../.venv/bin/python
   ```
   Or from the `flappy` directory with a custom Python:
   ```bash
   make PYTHON=/path/to/your/venv/bin/python
   ```

3. If raylib is not in `/opt/homebrew` or `/usr/local`, set:
   ```bash
   RAYLIB_INC="-I/path/to/raylib/include" RAYLIB_LIB="-L/path/to/raylib/lib -lraylib" make PYTHON=...
   ```

## Assets

Place `bird.png` and `pipe.png` in `resources/flappy/` (relative to the process CWD when running). The game uses them for rendering; run from the project root so `resources/flappy/` is found.

## Train / eval (v3)

From project root:

- **Train (Target-like policy/hparams):**
  `uv run python -m variations.flappyv3.train`
- **Train with custom timesteps:**
  `uv run python -m variations.flappyv3.train --train.total-timesteps 100000000`
- **Train with fixed non-curriculum difficulty (default 1.0):**
  `uv run python -m variations.flappyv3.train --env.fixed-difficulty 1.0`
- **Train with custom output dir:**
  `uv run python -m variations.flappyv3.train --train.output-dir variations/flappyv3/experiments_alt`
- **Eval with render:** `uv run python -m variations.flappyv3.run_eval --model path/to/model.pt`
- **Eval headless (stats):** `uv run python -m variations.flappyv3.run_eval --model path/to/model.pt --episodes 50 --no-render`
- **Batch eval last checkpoints:** `uv run python -m variations.flappyv3.eval_last_checkpoints --last 5 --episodes 50`

Default output location:

- `variations/flappyv3/experiments/<run_id>/model_XXXXXX.pt`
