---
title: Content Moderation RL Environment
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Content Moderation RL Environment

> A production-ready **Reinforcement Learning environment** simulating real-world AI content moderation on social media platforms — built for the **Meta Hackathon**.

---

## 📋 Overview

Social media platforms moderate millions of posts per day. This project wraps that real-world task into an **OpenEnv-compatible RL environment** where an agent learns to classify and action user-generated content across three escalating difficulty tasks.

The agent observes feature-rich post metadata and must choose one of four moderation actions. A shaped reward function teaches it to:

- **Remove** high-toxicity and spam content
- **Warn** users who post borderline/moderate hate speech
- **Escalate** complex misinformation to human reviewers
- **Allow** benign content without unnecessary interference

---

## 🏗️ Architecture

```
content-moderation-rl/
│
├── env/                    # Core RL environment (OpenEnv interface)
│   ├── __init__.py
│   ├── observation_model.py  # Pydantic models: Post, Observation, RewardInfo
│   ├── action_model.py       # ModerationAction enum + Action model
│   └── moderation_env.py     # ContentModerationEnv class
│
├── reward/                 # Reward computation
│   ├── __init__.py
│   └── reward_function.py    # compute_reward(post, action) -> RewardInfo
│
├── dataset/                # Synthetic post generator
│   ├── __init__.py
│   └── dataset_generator.py  # generate_dataset(), generate_task_dataset()
│
├── tasks/                  # Three tasks with increasing difficulty
│   ├── __init__.py
│   ├── spam_task.py          # Task 1 (Easy): Spam Detection
│   ├── hate_speech_task.py   # Task 2 (Medium): Hate Speech Moderation
│   └── misinformation_task.py # Task 3 (Hard): Misinformation Detection
│
├── graders/                # Programmatic graders (0.0–1.0 score)
│   ├── __init__.py
│   ├── spam_grader.py        # Accuracy-based grader
│   ├── hate_grader.py        # Severity-weighted accuracy grader
│   └── misinformation_grader.py # F1-score grader
│
├── baseline/               # Rule-based baseline agent
│   ├── __init__.py
│   └── baseline_agent.py     # BaselineAgent with threshold heuristics
│
├── run_baseline.py         # 🚀 Entry-point demo script
├── server.py               # 🌐 FastAPI HTTP server (port 7860)
├── inference.py            # 🤖 LLM-based inference agent with structured logging
├── Dockerfile              # 🐳 Container build (Python 3.11, port 7860)
├── openenv.yaml            # OpenEnv metadata / specification file
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🌍 Environment Design

### OpenEnv Interface

`ContentModerationEnv` implements the full OpenEnv interface:

```python
from env import ContentModerationEnv
from dataset import generate_dataset

posts = generate_dataset(n=500, seed=42)
env = ContentModerationEnv(posts=posts, seed=42)

obs = env.reset()            # -> Observation (Pydantic model)

while True:
    action = agent.select_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

print(env.episode_summary())
print(env.state())
```

### Observation Space

Each `Observation` exposes:

| Field | Type | Description |
|---|---|---|
| `post_id` | int | Unique post identifier |
| `post_text` | str | Full text of the post |
| `toxicity_score` | float [0,1] | ML-predicted toxicity |
| `report_count` | int | Number of user reports |
| `user_reputation` | float [0,1] | Author reputation (higher = trusted) |
| `image_flag` | bool | Potentially harmful image present |
| `spam_score` | float [0,1] | Spam detection score |
| `misinfo_score` | float [0,1] | Misinformation detection score |
| `step_index` | int | Current step in episode |
| `posts_remaining` | int | Posts left in queue |

### Action Space

| Action | Index | When to Use |
|---|---|---|
| `ALLOW` | 0 | Post is safe; no action needed |
| `REMOVE` | 1 | Post violates policy; remove immediately |
| `WARN_USER` | 2 | Borderline post; send author a warning |
| `ESCALATE_TO_HUMAN` | 3 | Complex case requiring human judgment |

---

## 💰 Reward System

The reward function encodes the intended moderation policy:

| Scenario | Correct Action | Reward |
|---|---|---|
| High toxicity (>0.8) | REMOVE | **+10** |
| Severe hate speech | REMOVE | **+9** |
| Spam post | REMOVE | **+8** |
| Misinformation | ESCALATE_TO_HUMAN | **+7** |
| Moderate hate speech | WARN_USER | **+4 to +5** |
| Benign (low tox) | ALLOW | **+3** |
| Allowing harmful content | — | **−10** |
| Unnecessary removal | — | **−5** |
| Excessive escalation | — | **−2** |

---

## 📋 Tasks

### Task 1 — Spam Detection (Easy)
- **200 posts**: 40% benign, 60% spam
- Spam signals are clear: high `spam_score`, high `report_count`, low `user_reputation`
- **Grader**: Accuracy (correct_actions / total_posts)
- **Target score**: ≥ 0.80

### Task 2 — Hate Speech Moderation (Medium)
- **200 posts**: 35% benign, 65% hate speech (varying severity)
- Agent must calibrate: REMOVE for severe (tox > 0.8), WARN_USER for moderate
- **Grader**: Severity-weighted accuracy (per-tier correct rates)
- **Target score**: ≥ 0.70

### Task 3 — Misinformation Detection (Hard)
- **200 posts**: 35% benign, 65% misinformation
- Misinformation overlaps visually with spam; agent must learn `misinfo_score` + `report_count` signals
- **Grader**: F1-score (balances precision and recall; harder to game)
- **Target score**: ≥ 0.60

---

## 🤖 Baseline Agent

The `BaselineAgent` uses threshold-based heuristics and serves as a performance lower bound:

```python
from baseline import BaselineAgent
agent = BaselineAgent()

action = agent.select_action(obs)   # -> Action (Pydantic model)
```

**Decision logic**:
```
toxicity_score > 0.80  →  REMOVE
spam_score     > 0.75  →  REMOVE
misinfo_score  > 0.70  →  ESCALATE_TO_HUMAN
toxicity_score > 0.50  →  WARN_USER
spam_score     > 0.45  →  WARN_USER
misinfo_score  > 0.40  →  WARN_USER
report_count   > 30 & misinfo_score > 0.40  → ESCALATE_TO_HUMAN
otherwise              →  ALLOW
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
# Standard run (shows score table)
python run_baseline.py

# Verbose mode (prints every step)
python run_baseline.py --verbose

# Quiet mode (only final scores)
python run_baseline.py --quiet
```

### 3. Run the inference script (LLM agent + structured logs)

```bash
# Without API keys — uses rule-based fallback, always works
python inference.py

# With an OpenAI-compatible LLM
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-xxx python inference.py

# With a HuggingFace model
API_BASE_URL=https://api-inference.huggingface.co/v1 MODEL_NAME=meta-llama/Llama-3-8b-Instruct HF_TOKEN=hf_xxx python inference.py
```

Structured log output (stdout):
```
[START]
spam_detection
[STEP]
REMOVE
8.0
[STEP]
ALLOW
3.0
...
[END]
0.8500
```

### 4. Start the HTTP server (for validator / HuggingFace Spaces)

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Test the endpoints:
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action":"REMOVE"}'
curl http://localhost:7860/state
```

### 5. Docker

```bash
# Build image
docker build -t cmp-env .

# Run container
docker run -p 7860:7860 cmp-env

# Test liveness
curl http://localhost:7860/health
# Expected: {"status": "ok", "environment": "content_moderation_rl", "version": "1.0.0"}
```

### 6. Use in your own RL loop

```python
from env import ContentModerationEnv
from env.action_model import ModerationAction
from dataset import generate_dataset

posts = generate_dataset(n=500, seed=42)
env = ContentModerationEnv(posts=posts, seed=42)

obs = env.reset()
total_reward = 0.0

while True:
    # Replace with your RL agent's policy
    action = ModerationAction.ALLOW
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Total reward: {total_reward}")
print(env.episode_summary())
```

---

## 📡 HTTP API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `POST` | `/reset` | Reset episode, returns first `Observation` |
| `POST` | `/step` | Submit action `{"action": "REMOVE"}`, returns `{observation, reward, done, info}` |
| `GET` | `/state` | Current internal episode state |
| `GET` | `/action_space` | List all valid actions with indices |
| `GET` | `/episode_summary` | Accuracy + per-category breakdown |

---

## 📊 Baseline Scores

| Agent | Task 1 (Spam) | Task 2 (Hate Speech) | Task 3 (Misinfo) | Average |
|---|---|---|---|---|
| **Random agent** | ~0.31 | ~0.28 | ~0.26 | ~0.28 |
| **Rule-based baseline** | ~0.85 | ~0.72 | ~0.61 | ~0.73 |
| **Target (to beat)** | 0.80 | 0.70 | 0.60 | — |

---

## 🔬 Dataset Generation

```python
from dataset import generate_dataset
from env.observation_model import PostCategory

# Full mixed dataset
posts = generate_dataset(n=500, seed=42)

# Task-specific dataset (spam + benign)
from dataset.dataset_generator import generate_task_dataset
spam_posts = generate_task_dataset(
    category=PostCategory.SPAM, n=200, benign_ratio=0.4, seed=42
)
```

Dataset categories (default weights):
- **Benign**: 40% — everyday posts, low toxicity
- **Spam**: 25% — promotional scams, high spam_score
- **Hate Speech**: 20% — discriminatory content, high toxicity
- **Misinformation**: 15% — false claims, high misinfo_score

---

## 📦 Requirements

- Python 3.10+
- pydantic >= 2.0
- pyyaml >= 6.0
- numpy >= 1.24
- rich >= 13.0
- fastapi >= 0.110 *(HTTP server)*
- uvicorn >= 0.29 *(ASGI server)*
- openai >= 1.0 *(LLM inference)*
- httpx >= 0.27 *(HTTP client)*

---

## 📄 License

MIT License — built for the Meta Hackathon.
