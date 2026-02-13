# PufferLib Target baseline

Train the **Target** sample environment from PufferLib’s ocean suite to compare convergence and step count vs your Flappy curriculum (e.g. 150M steps → ~15 pipes).

## What is Target?

From `target.h`: a small **multi-agent** env where “puffers” move in 2D and eat “stars” (goals). Each agent gets:

- **Obs:** normalized vectors to each goal, to each other agent, heading, last reward, and position (float box).
- **Actions:** `MultiDiscrete([9, 5])` — turn (9) and speed (5).
- **Reward:** +1 for reaching a goal (within 32 px); goals respawn randomly.

So it’s a simple foraging task with multiple agents and multiple goals. Good baseline to see how many steps and how fast PufferLib converges on a known env.

## How to train

From the **repo root**:

```bash
uv run python puffer_target_baseline/train_target.py
```

All checkpoints and trainer state go to:

- **`puffer_target_baseline/experiments/`**

so they don’t mix with your main `experiments/` or Flappy runs.

### Options

- Shorter run (e.g. 20M steps):
  ```bash
  uv run python puffer_target_baseline/train_target.py --train.total-timesteps 20_000_000
  ```
- Any PufferLib train/vec/env option can be overridden with `--section.key value` (e.g. `--train.learning-rate 0.01`). Default config: `config/ocean/target.ini` (total_timesteps 100M, LSTM policy, 512 envs, 8 agents, 4 goals).

## Default config (target.ini)

- **total_timesteps:** 100_000_000  
- **Policy:** Policy + Recurrent (LSTM)  
- **vec:** 512 envs (tuned down on CPU in the script if needed)  
- **env:** num_agents=8, num_goals=4 per env  

You can compare steps-to-convergence and wall-clock time vs your Flappy curriculum to see if Flappy is “hard” or just needs more tuning/time.
