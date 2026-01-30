# Quick orientation: Gymnasium + PufferLib

Short version so you know what’s going on without drowning in docs.

---

## 1. Gymnasium in 60 seconds

**What it is:** A standard API for “reinforcement learning environments.” Everyone (OpenAI, Farama, PufferLib, etc.) uses it or something compatible so code can be swapped.

**An “environment” is just a black box** that:

- **Reset:** Start (or restart) a run. You get an **observation** (what the agent “sees”) and maybe some **info**.
- **Step:** You send an **action**. You get back: new observation, **reward**, whether the episode **ended** (terminated or truncated), and maybe **info**.

So the loop is:

```
obs, info = env.reset()
while not done:
    action = policy(obs)   # your agent picks an action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

**Spaces** tell the library (and your agent) the shape of observations and actions:

- **Observation space:** e.g. `Box(low=-1, high=1, shape=(2,))` → “observation is 2 floats in [-1, 1]”.
- **Action space:** e.g. `Discrete(2)` → “action is 0 or 1”.

You implement `reset` and `step` and define those two spaces. That’s the whole contract. Everything else (training, batching, vectorization) is built on top of that.

**In this repo:** `SampleGymnasiumEnv` in `src/curly_succotash/env.py` is a minimal Gymnasium env: it has those spaces and stub `reset`/`step`. Run `uv run python -m curly_succotash` to see it used once.

---

## 2. What PufferLib actually is

**One sentence:** PufferLib is a library that **runs lots of copies of your env in parallel** and **trains an agent** on that data, with a focus on speed (millions of steps per second) and a small, readable training loop.

So you have two jobs:

1. **Define the env** (Gymnasium or PufferLib’s native “PufferEnv” format).
2. **Plug it into PufferLib** so it can vectorize (many envs) and train (PuffeRL).

PufferLib doesn’t replace Gymnasium. It **wraps** Gymnasium (and PettingZoo) envs so they can be:

- **Vectorized** — many envs stepped in parallel (sync or async).
- **Trained** — with PuffeRL (their PPO-style algorithm) or your own script.

So the stack looks like:

```
Your env (Gymnasium or PufferEnv)
    → PufferLib wraps / vectorizes it
        → PuffeRL (or you) trains a policy on batches of (obs, action, reward, ...)
```

**Important bits of PufferLib:**

| Thing | What it does |
|-------|----------------|
| **Emulation** | Wraps a Gymnasium env so PufferLib can batch and flatten observations/actions. One line: `GymnasiumPufferEnv(your_env)`. |
| **Vectorization** | Runs N copies of the env (e.g. 64) and gives you batches of N observations. Needed for fast training. |
| **PuffeRL** | Their training algorithm (PPO-style). You give it a vectorized env and a policy; it collects experience and updates the policy. |
| **CLI** | `puffer train <env>`, `puffer eval <env>` for built-in/Ocean envs. Your custom env you typically run via Python (e.g. `pufferl.train(...)` or a small script). |

You don’t have to understand the internals. For “get something working,” you only need:

1. A Gymnasium env (or a PufferEnv).
2. Wrap it with `GymnasiumPufferEnv` if it’s Gymnasium.
3. Call `pufferlib.vector.make(env_creator, num_envs=...)` to get a vectorized env.
4. Pass that plus a policy into PuffeRL (or use their `pufferl.train` with a custom loader).

---

## 3. What to do next

- **Just want to see the loop:** Run `uv run python -m curly_succotash` and look at `src/curly_succotash/env.py`. The Gymnasium part is the class with `observation_space`, `action_space`, `reset`, and `step`.
- **Want to train something:** Use a built-in env first so you see the full pipeline:  
  `uv run puffer train puffer_breakout --help`  
  Then try swapping in your own env via `pufferl.train(..., vecenv=your_vecenv, policy=...)` or the examples in [PufferLib examples](https://github.com/PufferAI/PufferLib/tree/3.0/examples) (e.g. `pufferl.py`).
- **Want the full API and env-writing details:** `ENV_INSTRUCTIONS.md` in this repo and the official docs: [puffer.ai/docs.html](https://puffer.ai/docs.html).

**TL;DR:** Gymnasium = “env with reset/step and spaces.” PufferLib = “vectorize that env and train an agent on it, fast.” Your job is to implement the env; PufferLib does the rest.

**About the “Gym has been unmaintained” message:** PufferLib supports the current Gymnasium API. That message is emitted by Gymnasium when it sees the old `gym` package installed (e.g. from PufferLib’s optional env extras, which still depend on `gym` for some third‑party envs). It’s harmless; you’re using Gymnasium correctly.
