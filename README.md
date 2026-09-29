# QLoRA Fine-Tuning Demo: HTML → JSON Extraction

Fine-tune a small open model to read a product HTML snippet and return a fixed JSON
object. The demo covers the full loop: prepare data, adapt a base model with QLoRA,
train, evaluate on held-out examples, export to GGUF, and serve it with Ollama.

The base model is **Qwen3-1.7B**, chosen so the whole run fits on one modest GPU and
finishes in minutes. [QLoRA](https://arxiv.org/abs/2305.14314) loads the frozen base model
in 4-bit and trains only small LoRA adapters, which is what keeps memory and cost down.

The task is narrow and the output schema is fixed. That is the case where fine-tuning a
small model beats prompting a large one: the model learns the mapping instead of being
re-told it in every prompt.

---

## What the model learns

Input (raw HTML):

```html
Extract the product information:
<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>
```

Output (JSON):

```json
{"name": "iPad Air", "price": "$1344", "category": "audio", "manufacturer": "Dell"}
```

---

## Repository layout

| File | What it is |
|---|---|
| `fine-tune.ipynb` | The main notebook: data prep → QLoRA → training → evaluation → GGUF export |
| `json_extraction_dataset_500.json` | 500 HTML → JSON training examples |
| `Modelfile` | Ollama definition for the exported GGUF model |
| `serve.sh` | Builds the Ollama model and exposes it through a Cloudflare quick tunnel |
| `pyproject.toml` | Project metadata (the training itself runs in the notebook) |

---

## The dataset

`json_extraction_dataset_500.json` is a list of 500 objects. Each one has:

- **`input`** — a prompt plus an HTML product snippet (`<h2>` name, `.price`,
  `.category`, `.brand`).
- **`output`** — the target object with exactly four keys: `name`, `price`, `category`,
  `manufacturer`.

Inputs and outputs are already paired, so this is a supervised fine-tuning set: show the
model the HTML, ask for the JSON, and measure the distance to the target. Over many steps
the model learns the mapping and the output schema.

The notebook trains on the first 450 rows and keeps the last 50 for evaluation, so the
score in Step 9 comes from data the model never saw.

---

## The walkthrough

Each step below is one cell in `fine-tune.ipynb`, in order.

### Step 1 — Load the dataset

Reads `json_extraction_dataset_500.json` and prints one example so you can see the shape
of the data. The file must be in the working directory.

### Step 2 — Install dependencies

Installs Unsloth plus matching versions of `transformers`, `trl`, `peft`, `accelerate`
and `bitsandbytes`. The versions are pinned because Unsloth patches the training stack at
import time; small mismatches can break a run.

### Step 3 — Check the environment

Confirms CUDA and a GPU are visible before loading the model. Failing here is cheaper than
failing halfway through a training run.

### Step 4 — Load the base model in 4-bit (QLoRA)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
```

- **`load_in_4bit=True`** quantizes the frozen base weights to 4-bit, cutting memory use
  roughly 4×. Unsloth uses 4-bit **NF4** quantization with double quantization, equivalent
  to:
  ```python
  BitsAndBytesConfig(load_in_4bit=True,
                     bnb_4bit_quant_type="nf4",
                     bnb_4bit_use_double_quant=True)
  ```
  This is the "Q" in QLoRA.
- **`max_seq_length=2048`** caps the context. These prompts are short.
- **`get_chat_template(..., chat_template="qwen3-instruct")`** installs the model's native
  ChatML template, so training and inference use the same prompt format. If they differ,
  output quality drops.

### Step 5 — Build the training text and hold out an eval set

Each example becomes a three-message conversation: system (the task instruction) → user
(the HTML) → assistant (the JSON). The conversation is rendered to a single string with
`tokenizer.apply_chat_template(...)` and stored in a `text` column. Training continues
that text, so the model learns to produce the assistant JSON given the first two turns.

The last 50 rows are kept out of training for the evaluation in Step 9.

### Step 6 — Attach LoRA adapters (the LoRA half of QLoRA)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    ...
)
```

Instead of updating all ~1.7B weights, **LoRA** freezes the 4-bit base model and trains
small low-rank matrices attached to the attention and MLP layers. Only a few million
parameters are trainable, which is why the run fits a modest GPU and produces a small
adapter file. `r=32` is the rank, or width, of those matrices. A 4-bit base plus LoRA
adapters is **QLoRA**.

### Step 7 — Configure the trainer

`SFTTrainer` runs the supervised fine-tuning loop. The knobs that matter:

| Setting | Value | Why |
|---|---|---|
| `per_device_train_batch_size` | 2 | Fits GPU memory |
| `gradient_accumulation_steps` | 4 | Effective batch size = 8 |
| `num_train_epochs` | 3 | Dataset is small (450 train rows) |
| `learning_rate` | 2e-4 | Typical QLoRA learning rate |
| `optim` | `adamw_8bit` | Memory-efficient optimizer |
| `lr_scheduler_type` | `linear` | Warm up, then decay to zero |

Then `train_on_responses_only(trainer)` masks the prompt tokens so the loss is computed
**only on the assistant's JSON**. Without it, the model is also trained to reproduce the
user's HTML, which wastes capacity.

### Step 8 — Train

`trainer.train()` runs the loop and prints wall-clock time and peak VRAM. On a 1.7B model
this finishes in a few minutes.

### Step 9 — Evaluate on held-out examples

Training loss only says the model is fitting. This step scores it on the 50 held-out rows:

- **Valid JSON** — the reply parses as JSON at all.
- **Exact match** — all four keys match the target.
- **Per-field accuracy** — each key on its own, so you can see which field is weak.

Decoding is greedy (`do_sample=False`), so the number is deterministic and comparable
between runs. Example output:

```
Held-out examples: 50
Valid JSON:        100.0%
Exact match:       94.0%
  name          98.0%
  price         100.0%
  category      98.0%
  manufacturer  96.0%
```

A high exact-match rate means the model learned the mapping, not just the JSON format. If
it is low, add epochs or raise LoRA rank `r` before touching the prompt.

### Step 10 — Try one prompt

Runs the model on a single HTML snippet. The chat template is applied with
`add_generation_prompt=True`, which appends the assistant header so the model starts
writing JSON. Sampling uses Qwen3-Instruct's recommended `temperature=0.7, top_p=0.8,
top_k=20`. Step 9 uses greedy decoding only because a benchmark should be repeatable.

### Step 11 — Export to GGUF

```python
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
```

**GGUF** is the single-file format used by Ollama and `llama.cpp`. `q4_k_m` is a good
accuracy-vs-size trade-off (~1.1 GB for a 1.7B model). The file is written to
`gguf_model_gguf/`, and the cell prints its exact path. Copy the `.gguf` into this folder
and keep the `Modelfile` `FROM` line pointed at it.

### Step 12 — Serve it with Ollama

Build and run the model directly:

```bash
ollama create qwen3-json-extractor -f Modelfile
ollama run qwen3-json-extractor "Extract the product information:
<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>"
```

The `Modelfile` sets the ChatML template, the sampling defaults and the system prompt, so
the local model behaves like the one you trained.

---

## Serving over a public URL with `serve.sh`

`serve.sh` builds the model in Ollama and exposes its API on a public `trycloudflare.com`
URL through a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/).
No Cloudflare account is needed.

```bash
./serve.sh
```

It will:

1. Auto-install `ollama`, `cloudflared` and system dependencies like `zstd` if they are
   missing.
2. Read the `Modelfile`'s `FROM` line and verify that GGUF file is present.
3. Start the Ollama server if it is not already running, then create the model.
4. Open a tunnel, rewriting the `Host` header to a localhost value so Ollama accepts the
   request, and print the public URL.

Call the model through the tunnel with Ollama's own API:

```bash
TUNNEL=https://<random>.trycloudflare.com
curl "$TUNNEL/api/generate" -d @- <<'JSON'
{
  "model": "qwen3-json-extractor",
  "prompt": "Extract the product information:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>",
  "stream": false
}
JSON
```

or through the OpenAI-compatible endpoint:

```bash
curl "$TUNNEL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-json-extractor",
       "messages": [{"role": "user", "content": "Extract the product information ..."}]}'
```

Override the defaults if needed:

```bash
MODEL=my-model PORT=11434 ./serve.sh
```

> **Security:** the quick tunnel has no authentication. Anyone with the URL can query the
> model. Ollama stays bound to `127.0.0.1`; only the tunnel is public. Stop the script with
> `Ctrl+C` to close it.

---

## Key concepts in one line each

- **Supervised fine-tuning (SFT)** — keep training a pretrained model on input→output
  pairs so it learns a specific task.
- **LoRA** — freeze the base model and train small low-rank adapter matrices instead.
- **QLoRA** — LoRA on top of a 4-bit-quantized base model, to save memory.
- **Chat template** — the exact token format (`<|im_start|>user ...`) the model expects.
  Training and inference must use the same one.
- **Response-only loss** — mask the prompt so only the answer contributes to the loss.
- **Held-out set** — examples kept out of training so the score is honest.
- **GGUF** — quantized single-file format for local inference (Ollama / llama.cpp).

---

## Concepts behind the steps

Every step applies an idea that predates this demo. This maps each step to the reason it
exists and a source for the idea.

### Step 1 — Load the dataset

**Purpose.** Supervised fine-tuning learns only from (input, target) pairs, so the dataset
is the entire supervision signal.

**Source.** Ouyang et al., *Training language models to follow instructions with human
feedback* (InstructGPT) — <https://arxiv.org/abs/2203.02155>

### Step 2 — Install dependencies

**Purpose.** Unsloth patches the model and training stack at import time, so a run is only
reproducible with a known-good set of versions.

**Source.** Unsloth documentation — <https://docs.unsloth.ai>

### Step 3 — Check the environment

**Purpose.** The 4-bit kernels need a CUDA GPU. Checking up front turns a failure that
would appear minutes into training into an immediate one.

**Source.** PyTorch, *CUDA semantics* — <https://pytorch.org/docs/stable/notes/cuda.html>

### Step 4 — Load the base model in 4-bit

**Purpose.** NF4 quantization with double quantization stores the frozen base weights in
about 4 bits while keeping fine-tuning quality close to a 16-bit base.

**Source.** Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* —
<https://arxiv.org/abs/2305.14314>

### Step 5 — Build the training text

**Purpose.** Models read tokens, not roles. Rendering the conversation through the model's
own chat template makes the training tokens identical in form to the inference tokens.

**Source.** Hugging Face, *Chat templates* —
<https://huggingface.co/docs/transformers/chat_templating>

### Step 6 — Attach LoRA adapters

**Purpose.** LoRA freezes the base weights and learns a low-rank update, which reduces the
trainable parameter count by orders of magnitude with little quality loss.

**Source.** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* —
<https://arxiv.org/abs/2106.09685>

### Step 7 — Configure the trainer

**Purpose.** Instruction tuning computes loss on the answer, not the question, so
`train_on_responses_only` masks the prompt. `adamw_8bit` quantizes optimizer state
block-wise to cut memory further.

**Sources.** Ouyang et al. (response-only loss) — <https://arxiv.org/abs/2203.02155>;
Dettmers et al., *8-bit Optimizers via Block-wise Quantization* —
<https://arxiv.org/abs/2110.02861>; TRL `SFTTrainer` —
<https://huggingface.co/docs/trl/sft_trainer>

### Step 8 — Train

**Purpose.** The loop minimizes the masked next-token loss. Gradient accumulation sums
gradients over several micro-batches, so the effective batch size grows without growing
memory.

**Sources.** Hugging Face, *Methods and tools for efficient training on a single GPU* —
<https://huggingface.co/docs/transformers/perf_train_gpu_one>; Chen et al., *Training Deep
Nets with Sublinear Memory Cost* (gradient checkpointing) — <https://arxiv.org/abs/1604.06174>

### Step 9 — Evaluate on held-out examples

**Purpose.** Training loss only measures fit. A held-out set estimates generalization, and
exact match is the strict metric for structured extraction.

**Source.** scikit-learn, *Cross-validation: evaluating estimator performance* —
<https://scikit-learn.org/stable/modules/cross_validation.html>

### Step 10 — Try one prompt

**Purpose.** Generation uses the same template minus the answer, with `add_generation_prompt`.
Top-p sampling truncates the tail of the distribution, which avoids the repetition that
pure greedy decoding produces on open-ended text.

**Sources.** Qwen, *Qwen3-1.7B model card* (recommended sampling) —
<https://huggingface.co/Qwen/Qwen3-1.7B>; Holtzman et al., *The Curious Case of Neural Text
Degeneration* — <https://arxiv.org/abs/1904.09751>

### Step 11 — Export to GGUF

**Purpose.** GGUF is a single file holding the model metadata and quantized tensors that
llama.cpp and Ollama read. `q4_k_m` mixes 4-bit types per tensor for a size/quality
trade-off.

**Sources.** GGUF specification —
<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>; Hugging Face, *GGUF* —
<https://huggingface.co/docs/hub/gguf>

### Step 12 — Serve it with Ollama

**Purpose.** Ollama loads the GGUF and serves a local API. A `Modelfile` records the
template and sampling defaults so the served model matches training, and a quick tunnel
puts that local API on a public HTTPS URL.

**Sources.** Ollama, *Modelfile* — <https://docs.ollama.com/modelfile>; Cloudflare,
*Quick tunnels* —
<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/>

---

## Requirements

- A CUDA-capable GPU. The 1.7B model is trained in 4-bit to fit modest hardware.
- Python with internet access to install dependencies and download the base model. The
  notebook installs everything it needs.

To run the notebook: open `fine-tune.ipynb` in Jupyter, make sure
`json_extraction_dataset_500.json` is in the working directory, and run the cells in
order. Step 11 exports the GGUF.

---

## Customizing the demo

### Try another base model

Change the `model_name` string in Step 4. Small options:
`unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (default),
`unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (smallest), or
`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`.

If the model changes family, change the chat template in the same cell (for example
`llama-3.1` for Llama) and re-point the `Modelfile` `FROM` line.

### Other ideas

- Raise `num_train_epochs` or LoRA rank `r` if Step 9 shows the model underfits.
- Constrain decoding with a JSON schema or grammar so output is always valid.
- Push the adapter or GGUF to the Hugging Face Hub with `push_to_hub_gguf`.
