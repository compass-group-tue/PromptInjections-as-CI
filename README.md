# AI Agents May Always Fall for Prompt Injections

This repository contains datasets, evaluation code, and results for the paper:

> **AI Agents May Always Fall for Prompt Injections**  
> *Anonymous Authors*  
> Under review.

---

## Overview

We argue that prompt injection is most naturally understood as a violation of **Contextual Integrity (CI)** — a privacy theory that judges information flow compliance with contextual norms. CI decomposes the appropriateness of an agent's action along five dimensions: *sender*, *receiver*, *subject*, *information type*, and *transmission principle*. This reframing exposes why current defenses (data-instruction separation, injection classifiers, alignment training) fail against contextually grounded attacks, and predicts the attack surface that future autonomous agents will face.

The paper presents four empirical contributions, each corresponding to a directory in this repository:

| Directory | Paper Section | Description |
|---|---|---|
| [`defense_evaluation/`](#defense_evaluation) | §3 | Limitations of current defenses: injection classifiers and SecAlign safety training on a paired email dataset |
| [`ci_redteaming/`](#ci_redteaming) | §5.1 | CI-grounded red-teaming: context parameter attacks (96.7% ASR) and norm-evaluation attacks |
| [`norm_grounding/`](#norm_grounding) | §5.2 | Norm inference degrades without grounding history |
| [`flow_separation/`](#flow_separation) | §5.3 | Simultaneous information flows: authorization for one flow leaks into another |

---

## Repository Structure

```
.
├── defense_evaluation/          # §3 — Paired email dataset & defense limitations
│   ├── paired_emails_dataset.json #  4,200 paired email scenarios (attack + benign)
│   ├── llmail_inject_dataset.json #  LLMail-Inject scenarios (security alert evaluation)
│   ├── generate_data/           #   Dataset generation scripts
│   │   ├── generate_dataset.py  #   Generate paired email scenarios with an LLM
│   │   └── generation_prompt.txt#   Prompt template for dataset generation
│   ├── classifier_eval/         #   Injection classifier evaluation (PromptGuard 1 & 2)
│   │   ├── eval_classifier.py
│   │   └── results/
│   ├── agentic_task/            #   Email assistant agent evaluation
│   │   ├── eval_agent.py        #   Main evaluation script
│   │   ├── model_client.py      #   Unified API client (Anthropic, OpenAI, Azure, vLLM)
│   │   ├── run_eval.sh          #   Run frontier model evaluation
│   │   ├── run_eval_secalign.sh #   Run SecAlign-70B evaluation (vLLM)
│   │   ├── run_eval_llama_baseline.sh  # Run LLaMA-3.3-70B baseline (vLLM)
│   │   └── results/
│   └── flagging/                #   Security alert detection evaluation (LLMail-Inject)
│       ├── eval_detection.py    #   Evaluate LLaMA-3.3-70B and SecAlign-70B for alert flagging
│       ├── inspect_alerts.py    #   Print uncaught / vacuous alert strings for filter tuning
│       ├── compare_checkpoints.py   #   Side-by-side table: send_email rate + detection rate for both models
│       ├── run_eval_baseline.sh #   Run LLaMA-3.3-70B baseline evaluation
│       ├── run_eval_secalign.sh #   Run Meta SecAlign-70B evaluation (via vLLM)
│       └── results/             #   Evaluation outputs (JSON)
│
├── ci_redteaming/               # §5.1 — CI-grounded red-teaming attacks
│   ├── parameter_attacks/       #   PAIR-style context parameter attacks
│   │   ├── pair_attack.py       #   Iterative red-team loop (adapts PAIR framework)
│   │   ├── baseline_eval.py     #   Evaluate unoptimized email baseline
│   │   ├── transferability_eval.py  # Evaluate attack transfer to other models
│   │   ├── extract_best_attacks.py  # Extract best-performing attacks per scenario
│   │   ├── run_pair.sh
│   │   ├── baseline_eval_run.sh
│   │   ├── transferability_eval_run.sh
│   │   └── results/             #   Attack outputs + transferability results
│   └── norm_evaluation_attacks/ #   Value-argument norm-evaluation attacks
│       ├── pair_value_manipulation.py   # Red-team loop constrained to value arguments
│       ├── validate_attacks.py          # Validate attack success and constraint adherence
│       ├── run_pair_value.sh
│       ├── validate_attacks.sh
│       └── results/
│
├── norm_grounding/              # §5.2 — Norm inference requires grounding history
│   ├── dataset.json             #   300 multi-turn scenarios (4 domains, 3 escalation types)
│   ├── generate_data/           #   Dataset generation
│   │   ├── generate_dataset.py  #   Generate norm-grounding scenarios with an LLM
│   │   ├── prompt.txt           #   Prompt template for dataset generation
│   │   ├── run_generate.sh      #   Run dataset generation
│   │   ├── edit_dataset.py      #   Post-process dataset (add tool calls, scope checks)
│   │   └── run_edit.sh          #   Run dataset editing step
│   └── agentic_task/            #   Agent evaluation with/without history
│       ├── eval_agent.py
│       ├── compute_rates.py     #   Compute execution rates from results
│       ├── run_eval.sh
│       └── results/
│
└── flow_separation/             # §5.3 — Simultaneous information flow separation
    ├── flow_separation_scenarios.json  # 100 two-flow email thread scenarios
    ├── eval_agent.py            #   Evaluate F1 utility + F2 security violation rates
    ├── run_eval.sh
    └── results/
```

---

## Setup

### Requirements

- Python 3.10+
- API keys for the model providers you use (Anthropic, OpenAI, Google Gemini, or Azure OpenAI)
- For local model evaluation (SecAlign, LLaMA baselines): a GPU cluster with [vLLM](https://github.com/vllm-project/vllm) installed

### Installation

```bash
git clone <repo-url>
cd PromptInjections-as-CI
pip install anthropic openai google-generativeai transformers vllm filelock tqdm
```

### API Keys

Set keys as environment variables before running any script:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export AZURE_OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export HF_TOKEN=...                  # for gated HuggingFace models (SecAlign)
```

Alternatively, the run scripts check for plaintext key files and load them automatically:

| Provider | Environment Variable | Key file (one token per file) |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `/path/to/anthropic_token` |
| OpenAI | `OPENAI_API_KEY` | `/path/to/openai_token` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `/path/to/azure_openai_token` |
| Google Gemini | `GEMINI_API_KEY` | `/path/to/gemini_key` |

To use key files, uncomment and edit the `if [ -f ... ]` block near the top of each `run_*.sh` script to point to your key file paths.

### Azure OpenAI Configuration

Scripts that call Azure OpenAI (dataset generation, some attack scripts) require two additional variables:

```bash
export AZURE_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com
export AZURE_DEPLOYMENT=gpt-4o          # or whichever deployment you have
export AZURE_API_VERSION=2025-01-01-preview
```

These can also be passed inline: `AZURE_ENDPOINT=... bash run_generate.sh`.

---

## Running Experiments

### §3 — Defense Evaluation (Paired Email Dataset)

**Generate the paired email dataset** (the generated dataset also exists):
This generates pairs of emails, one asserts an authority to reply, and one neutral. 
```bash
cd defense_evaluation/generate_data
AZURE_ENDPOINT=<url> AZURE_DEPLOYMENT=<deployment> python generate_dataset.py --num-items 4200 --output ../paired_emails_dataset.json
```

**Security-alert detection evaluation** (LLMail-Inject scenarios, `flagging/`):

This experiment asks whether a model functioning as an email assistant will *flag* injected instructions as suspicious — i.e., produce a security alert in its response — rather than silently ignore them. We use the [LLMail-Inject](https://github.com/microsoft/LLMail-Inject) benchmark scenarios and compare two models:

- **Llama-3.3-70B-Instruct** (undefended baseline, loaded directly via vLLM)
- **Meta-SecAlign-70B** (LoRA adapter on the same base model, loaded via vLLM)

Each scenario is fed to the model as an email assistant task. The script records whether `send_email` was called (successful attacks) and whether the model correctly emitted a security alert (`injection_detected`).

Download model weights before running (see [Setup](#setup)):
```bash
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct  # base model for SecAlign adapter
huggingface-cli download facebook/Meta-SecAlign-70B
```

Evaluate LLaMA-3.3-70B-Instruct (undefended baseline):
```bash
cd defense_evaluation/flagging
bash run_eval_baseline.sh
# or with a limit for testing:
MAX_ITEMS=50 bash run_eval_baseline.sh
```

Evaluate Meta SecAlign-70B:
```bash
cd defense_evaluation/flagging
bash run_eval_secalign.sh
# or with a limit for testing:
MAX_ITEMS=50 bash run_eval_secalign.sh
```

Both scripts write a checkpoint file under `results/` as they run (every `BATCH_SIZE` items), so they can be interrupted and resumed.

**Inspect alert quality** — some `injection_detected=True` results contain vacuous alerts ("None found", "No suspicious content"). `inspect_alerts.py` prints some unique alert strings that are *not* caught by the built-in false-positive filter, so you can refine the filter if needed:
```bash
cd defense_evaluation/flagging
python inspect_alerts.py
```

**Compare baseline vs. SecAlign** — `compare_checkpoints.py` loads the checkpoint files from both runs and prints a side-by-side table of `send_email` rate, raw detection rate, false-positive corrections, and corrected detection rate:
```bash
cd defense_evaluation/flagging
python compare_checkpoints.py
```

> **Note:** `compare_checkpoints.py` and `inspect_alerts.py` have hard-coded result paths at the top of each file. Edit the `BASELINE_PATH` and `SECALIGN_PATH` variables to point to your result files before running.

**Classifier evaluation** (PromptGuard 1, PromptGuard 2):

We measure whether off-the-shelf prompt-injection classifiers can distinguish attack emails from benign ones in our paired dataset. Each of the 4,200 email pairs is scored by two classifiers — `meta-llama/Prompt-Guard-86M`, and `meta-llama/Llama-Prompt-Guard-2-86M` — and precision, recall, and F1 are reported. The key finding is that contextually grounded injections evade these two classifiers because they are indistinguishable from legitimate email text by shallow syntactic/semantic features alone.

Download the models first if not already cached:
```bash
huggingface-cli download meta-llama/Prompt-Guard-86M
huggingface-cli download meta-llama/Llama-Prompt-Guard-2-86M
```
Then run:
```bash
cd defense_evaluation/classifier_eval
bash run_eval.sh
# or with a limit for testing:
MAX_ITEMS=100 bash run_eval.sh
```

**Agent evaluation** (closed frontier models and SecAlign):

An LLM acts as an email assistant with a `send_email` tool. Each scenario gives the model a system prompt, the tool, a user message, and one email variant (attack or benign). The system prompt uses *conditional* delegation — it permits action when the user has authorized it or the request aligns with verifiable organizational policy — rather than blanket permission or blanket refusal. We test two variants of the system prompt (permissive and restrictive delegation) crossed with two email variants (authority-asserting and neutral) from the paired dataset.

For closed models:
```bash
cd defense_evaluation/agentic_task
BACKEND=anthropic MODEL=claude-sonnet-4-6 bash run_eval.sh
# Azure OpenAI:
BACKEND=azure MODEL=gpt-4o AZURE_ENDPOINT=<url> bash run_eval.sh
```

For SecAlign-70B and the LLaMA baseline (requires GPU, models downloaded first):
```bash
cd defense_evaluation/agentic_task
bash run_eval_secalign.sh
bash run_eval_llama_baseline.sh
# or with a limit for testing:
MAX_ITEMS=50 bash run_eval_secalign.sh
```

### §5.1 — Attacking Context Parameters Inference and Norm Evaluation

This experiment red-teams the CI agent's context parameter inference and norm-evaluation steps. **Context parameter attacks** use a PAIR-style iterative loop where an attacker LLM crafts emails designed to manipulate the agent into inferring wrong parameter values (e.g., an expanded recipient scope), guided by a judge that scores whether the target agent's `send_email` call violated the intended constraint. After optimization, `extract_best_attacks.py` selects the highest-scoring email per scenario and `transferability_eval.py` tests whether those attacks transfer to other target models. **Norm-evaluation attacks** use a similar loop but target the norm-evaluation step: the attacker generates value-based arguments embedded in emails to convince the agent that a norm violation is actually justified.

**Context parameter attacks** (PAIR-style red-teaming loop):
```bash
cd ci_redteaming/parameter_attacks
# Run PAIR attack: attacker=claude-sonnet-4-6, target=claude-sonnet-4-6
ATTACKER_BACKEND=anthropic ATTACKER_MODEL=claude-sonnet-4-6 \
TARGET_BACKEND=anthropic TARGET_MODEL=claude-sonnet-4-6 bash run_pair.sh

# Evaluate transferability to other models using best attacks
BACKEND=openai MODEL=gpt-5.4 bash transferability_eval_run.sh
```

**Norm-evaluation attacks** (value-argument manipulation):
```bash
cd ci_redteaming/norm_evaluation_attacks
# Attacker=claude-sonnet-4-6, target=gpt-5.2
ATTACKER_BACKEND=anthropic ATTACKER_MODEL=claude-sonnet-4-6 \
TARGET_BACKEND=openai TARGET_MODEL=gpt-5.2 bash run_pair_value.sh
```

### §5.2 — Norm Grounding

This experiment tests whether models can infer the correct norms for a task purely from conversation history (grounding), and whether removing that history degrades norm compliance. Each scenario is a multi-turn delegation conversation across four domains (legal, medical, financial, HR) with three escalation types. The agent is evaluated on whether it executes in-scope requests and correctly refuses out-of-scope ones. A `boundary_turn` can optionally be injected to make the scope boundary explicit; without it the agent must infer limits from context alone.

**Generate the norm-grounding dataset**:
```bash
cd norm_grounding/generate_data
AZURE_ENDPOINT=<url> AZURE_DEPLOYMENT=<deployment> bash run_generate.sh
# Then post-process (adds tool calls, scope checks, delegation responses):
AZURE_ENDPOINT=<url> AZURE_DEPLOYMENT=<deployment> bash run_edit.sh
```

Run agent evaluation with varying amounts of grounding history:
```bash
cd norm_grounding/agentic_task
BACKEND=anthropic MODEL=claude-sonnet-4-6 bash run_eval.sh
# With boundary turn:
BOUNDARY_TURN=1 BACKEND=anthropic MODEL=claude-sonnet-4-6 bash run_eval.sh

# Compute rates from results
python compute_rates.py --results results/
```

### §5.3 — Flow Separation

Each scenario contains a single email thread that carries *two simultaneous information flows*: F1 is an authorized request (e.g., the principal asking the agent to take an action) and F2 is an unauthorized one (e.g., a third-party email in the same thread trying to exfiltrate data via `send_email`). The agent must fulfill F1 while refusing F2. 

We evaluate two conditions: *no_boundary* (the agent receives the delegation statement only) and *with_boundary* (the agent also receives an explicit delegation boundary statement). The two metrics are **utility** (did the agent take the authorized F1 action?) and **security violation** (did `send_email` get called with an external contact's address, crossing the F2 boundary?).

```bash
cd flow_separation
BACKEND=anthropic MODEL=claude-sonnet-4-6 bash run_eval.sh
# With explicit boundary statement:
BACKEND=anthropic MODEL=claude-sonnet-4-6 bash run_eval.sh --boundary-turn
# or with a limit for testing:
MAX_ITEMS=20 BACKEND=anthropic MODEL=claude-sonnet-4-6 bash run_eval.sh
```

---

## Datasets

| Dataset | Location | Description |
|---|---|---|
| Paired email scenarios | `defense_evaluation/paired_emails_dataset.json` | 4,200 paired (attack/benign) email scenarios for an email assistant agent |
| LLMail-Inject scenarios | `defense_evaluation/llmail_inject_dataset.json` | LLMail-Inject scenarios used for security-alert detection evaluation |
| Norm grounding scenarios | `norm_grounding/dataset.json` | 300 multi-turn delegation scenarios across 4 domains, 3 escalation types |
| Flow separation scenarios | `flow_separation/flow_separation_scenarios.json` | 100 two-flow email thread scenarios across 50 professional domains |

---

## Results

Pre-computed results from the paper are provided under each `results/` directory. Result filenames encode the model and backend used, e.g., `eval_anthropic_claude-sonnet-4-6.json`.

---

## Citation

```bibtex
@article{anonymous2026promptinjectionasci,
  title   = {AI Agents May Always Fall for Prompt Injections},
  author  = {Anonymous Authors},
  year    = {2026},
  note    = {Under review}
}
```
