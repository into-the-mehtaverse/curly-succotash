# Flappy v2 curriculum + LSTM (C + raylib)

Copy of `variations/flappy` with a dedicated v2 training path that uses an LSTM policy.
All v2 checkpoints are isolated under `variations/flappyv2/experiments/`.

## Build

1. **Install raylib** (required for rendering):
   - macOS: `brew install raylib`
   - Linux: install `libraylib-dev` (or build from [raylib](https://www.raylib.com/)).

2. **Build the C extension** from repo root (use the venv Python so the .so loads):
   ```bash
  cd variations/flappyv2
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

## Train / eval (v2)

From project root:

- **Train (LSTM + curriculum):**
  `uv run python -m variations.flappyv2.train`
- **Train with custom timesteps:**
  `uv run python -m variations.flappyv2.train --train.total-timesteps 150000000`
- **Train with custom output dir:**
  `uv run python -m variations.flappyv2.train --train.output-dir variations/flappyv2/experiments_alt`
- **Eval with render:** `uv run python -m variations.flappyv2.run_eval --model path/to/model.pt`
- **Eval headless (stats):** `uv run python -m variations.flappyv2.run_eval --model path/to/model.pt --episodes 50 --no-render`
- **Batch eval last checkpoints:** `uv run python -m variations.flappyv2.eval_last_checkpoints --last 5 --episodes 50`

Default output location:

- `variations/flappyv2/experiments/<run_id>/model_XXXXXX.pt`
