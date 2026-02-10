---
title: "Building my first RL environment."
description: "Pt. 3 of building undeniable technical ability"
pubDate: 2026-02-03
---

I find reinforcement learning to be the most interesting sector of ML at the moment for its unique attributes of requiring less compute than supervised learning policies, relevance to the next paradigm of AI (robotics / world models), and for that it is largely neglected relative to LLMs and other forms of ML.

Today I built my first environment, namely a two-grid flappy bird where the agent has two actions, up or down, and receives a -1 reward for hitting the floor or roof, and 0 reward for staying alive.

My first version of the env just returns three values for the observation, namely, [position of agent, roof pos, floor pos].

**Mistakes and iterations.** Initially I didn’t include the agent’s position in the observation—only whether there was a wall on the roof or floor. The agent had to infer “am I at floor or ceiling?” from history, which made learning much harder. I fixed that. Reward was only -1 on death and 0 otherwise, so the signal was sparse; I added a small +0.01 per step survived so “stay alive” had a positive return. Training hit a few snags: the vectorizer calls `driver_env.close()`, and the base PufferEnv doesn’t implement it, so I added a no-op `close()`. Default config gave `batch_size < minibatch_size` with only two envs; I increased `num_envs` so the batch was large enough for PuffeRL.

**Why wasn’t it learning in 10 steps?** The rule is trivial: at roof don’t go up, at floor don’t go down. I expected the policy to overfit that in a handful of steps. What I got was entropy stuck near 0.69 (max for two actions) for 500k–2M steps. The reason isn’t the task—it’s the algorithm. RL doesn’t get “correct action” labels; it explores, gets rewards, and slowly reinforces better actions. PPO batches thousands of steps per update and clips policy changes, so each bad (or good) experience only nudges the policy a little. The setup is built for stability in hard envs, not “memorize this rule in 10 steps.”

**Making it learn faster.** I bumped the learning rate (3e-4 → 0.01) and loosened the PPO clip (0.2 → 0.5). The policy went deterministic (entropy → 0) within 2M steps. So the agent *can* learn the rule; it just needed a more aggressive update.

**Sweep: where does it become deterministic?** I added a small grid sweep over learning rate and clip coefficient, ran 500k steps per (lr, clip) pair, and recorded final entropy. Result: at lr = 0.03 the policy goes deterministic for every clip value tried (0.2, 0.35, 0.5, 0.7). At lr = 0.01 entropy drops to ~0.05–0.14 but doesn’t cross the “deterministic” threshold in 500k steps. At 0.001 and 0.003 the policy stays exploratory (entropy ~0.6). So LR is the main lever; clip_coef barely mattered in this grid.

Sweep summary (final entropy per lr × clip_coef, 500k steps per run):


| lr     | clip_coef | entropy | epoch | deterministic |
|--------|-----------|---------|-------|----------------|
| 0.001  | 0.2       | 0.6721  | 55    | no             |
| 0.001  | 0.35      | 0.6807  | 55    | no             |
| 0.001  | 0.5       | 0.6823  | 60    | no             |
| 0.001  | 0.7       | 0.6495  | 61    | no             |
| 0.003  | 0.2       | 0.6151  | 61    | no             |
| 0.003  | 0.35      | 0.6352  | 58    | no             |
| 0.003  | 0.5       | 0.6229  | 55    | no             |
| 0.003  | 0.7       | 0.5921  | 54    | no             |
| 0.01   | 0.2       | 0.0552  | 54    | no             |
| 0.01   | 0.35      | 0.1178  | 60    | no             |
| 0.01   | 0.5       | 0.1123  | 55    | no             |
| 0.01   | 0.7       | 0.1359  | 55    | no             |
| 0.03   | 0.2       | 0.0000  | 58    | yes            |
| 0.03   | 0.35      | 0.0000  | 60    | yes            |
| 0.03   | 0.5       | 0.0000  | 61    | yes            |
| 0.03   | 0.7       | 0.0000  | 60    | yes            |

---

**Flappy (C + raylib): reward, obs, and env iterations.** After porting to a proper Flappy Bird–style env (pipes, gap, continuous-ish physics), the agent often couldn’t get through the first pipe. Below are the changes I tried, what broke, and how things improved to where they are now.

**Env / physics tweaks**
- **Slower pipes** – Reduced `PIPE_SPEED_RATIO` from 0.012 → 0.006 so the bird had more time to line up. Without this, the agent rarely got +1 for passing a pipe and learning didn’t take off.
- **Lower flap velocity** – Started at 0.055; the bird overshot and hit the top pipe. I stepped it down (0.04 → 0.032 → 0.022 → 0.02). Each reduction gave finer control and less “flap into ceiling.” This was one of the biggest levers for actually getting through pipes.

**Rewards**
- **Survival bonus** – Already had +0.01 per step alive so the policy had a positive signal for “don’t die.”
- **In-gap bonus** – +0.02 when the bird is *inside* the gap, scaled by distance to the pipe (closer = more). Helps with “stay in the safe zone” once you’re there.
- **Alignment bonus** – Added a small reward for being *near* the gap center *before* entering the gap (e.g. 0.008 per step, decaying with vertical distance over a tolerance of 0.2). Goal: encourage lining up early instead of last-second flapping. This helped.
- **Streak bonus** – Pipe-pass reward became 1.0 for first pipe, 1.1 for second, 1.2 for third, etc., so the agent is incentivized to keep going for later pipes.
- **Flap penalty** – Tried a tiny cost per flap (0.001) to discourage flapping when already high. In practice it didn’t seem to change behavior much. I didn’t push it higher because **too much flap penalty makes the bird too passive and it deterministically hits the ground**—the policy stops flapping enough to stay in the air.
- **Penalty for being above the gap** – Considered a small negative reward when the bird is above the gap center to reduce “flap into top pipe.” Didn’t implement it because **strong “above gap” penalties also risk the bird preferring to fall and hit the ground** rather than risk being “too high.” Same failure mode as an oversized flap penalty.

**Observations**
- **Original 7-D** – bird y, bird vy, distance to next pipe, gap center, gap height, “is there a next pipe?”, and signed gap error (gap_center − bird_y, clamped to ±1). The clamp on gap error meant “slightly above” and “way above” both looked similar (both negative), so the policy didn’t get a clear “don’t flap, you’re way too high” signal.
- **Added clearance from top and bottom of gap (9-D)** – Two new dims: signed clearance from the *top* of the gap and from the *bottom*, in half-gap units, clamped to ±1. So the agent sees explicitly “how far am I from the top pipe?” and “how far am I from the bottom pipe?” That made “way above” vs “slightly above” learnable and helped with consistency on the first pipe and sometimes the second.

**Where things stand**
- With slower pipes, reduced flap velocity, alignment + streak rewards, a small flap penalty, and the extra clearance obs, the bird **reliably gets the first pipe and sometimes the second**. Episode length hovers around ~200 steps; more training (e.g. 40M steps) didn’t push it much past that—so we’re at a plateau. Perhaps a local minima that just gets the first pipe and doesnt really try for later ones. Trying different checkpoints (earlier saves from the same run) can sometimes yield a slightly more consistent policy than the final one.
