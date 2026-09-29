# QLoRA Fine-Tuning Demo — HTML → JSON Extraction

Fine-tune a small open-source LLM to read a product HTML snippet and return clean,
structured JSON.

This is an end-to-end demo of **supervised fine-tuning (SFT)** with **QLoRA** using
[Unsloth](https://github.com/unslothai/unsloth). QLoRA means the base model is loaded in
4-bit and only small LoRA adapters are trained, so the whole run fits on a single modest
GPU. The fine-tuned model is exported to **GGUF** so you can run it locally with
[Ollama](https://ollama.com) or `llama.cpp`.

The base model is **Qwen3-4B-Instruct-2507**, a small instruction-tuned model chosen
because it is cheap to fine-tune and run.

---

## Table of contents

- [Purpose](#purpose)
- [What the model learns to do](#what-the-model-learns-to-do)
- [Repository layout](#repository-layout)
- [The dataset](#the-dataset)
- [How the demo works (walkthrough)](#how-the-demo-works-walkthrough)
- [Serving with a Cloudflare tunnel](#serving-with-a-cloudflare-tunnel)
- [Key concepts in one line each](#key-concepts-in-one-line-each)
- [Getting started](#getting-started)
- [Customizing the demo](#customizing-the-demo)

---

## Purpose

The goal of this repo is to show, end to end and with nothing hidden behind a framework
abstraction, **how to fine-tune an LLM for a real task on hardware most people have
access to**. Specifically:

- Demonstrate the full fine-tuning lifecycle — prepare data, adapt a base model with
  QLoRA, train, evaluate by hand, and ship the result — on a single modest GPU.
- Show why fine-tuning beats prompting for a narrow, repetitive task with a fixed output
  schema: the model learns the HTML → JSON mapping, so you stop spending prompt tokens
  and a large model to do a small, well-defined job.
- Make the moving parts concrete: what 4-bit quantization, LoRA adapters, chat templates,
  response-only loss and GGUF export actually do, and where each one sits in the pipeline.
- Produce a portable artifact — a small `q4_k_m` GGUF file — that runs locally with Ollama
  or `llama.cpp`, with no cloud required.

It is written as a teaching demo: every step maps to one cell in `fine-tune.ipynb`, and
the companion sections below explain the reasoning behind it, not just the code.

---

## What the model learns to do

Input (raw HTML):

```html
Extract the product information:
<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>
```

Output (JSON):

```json
{"name": "iPad Air", "price": "$1344", "category": "audio", "manufacturer": "Dell"}
```

The task is narrow and the output schema is fixed, which is exactly the kind of job that
fine-tuning a small model does better, faster and cheaper than prompting a large one.

---

## Repository layout

| File | What it is |
|---|---|
| `fine-tune.ipynb` | The main notebook: data prep → QLoRA → training → inference → GGUF export |
| `json_extraction_dataset_500.json` | 500 HTML → JSON training examples |
| `Modelfile` | Ollama definition for the exported GGUF model |
| `serve.sh` | Builds the Ollama model and exposes it through a Cloudflare quick tunnel |
| `pyproject.toml` | Minimal project metadata (the training itself runs in the notebook) |

---

## The dataset

`json_extraction_dataset_500.json` is a list of 500 objects. Each one has:

- **`input`** — a prompt plus an HTML product snippet (`<h2>` name, `.price`,
  `.category`, `.brand`).
- **`output`** — the target object with exactly four keys: `name`, `price`, `category`,
  `manufacturer`.

Because inputs and outputs are already paired, this is a supervised fine-tuning dataset:
we show the model the HTML, ask for the JSON, and measure how far its answer is from the
target. Over many steps the model learns the mapping (and the output schema) rather than
being told it in the prompt every time.

---

## How the demo works (walkthrough)

The notebook is 11 short steps. Here is what each one does and why.

### 1. Load the dataset
Reads `json_extraction_dataset_500.json` into memory and prints one example so you can
see the shape of the data. Make sure the file is in the working directory first — step 3
of [Getting started](#getting-started) shows how to set it up.

### 2. Install dependencies
Installs Unsloth plus a compatible set of `transformers`, `trl`, `peft`, `accelerate`
and `bitsandbytes`. Versions are pinned because Unsloth patches the training stack and
small version mismatches can break the run.

### 3. Check the environment
Confirms that CUDA and a GPU are available before loading the model and training.

### 4. Load the base model in 4-bit (QLoRA)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=2048,
    load_in_4bit=True,
)
```
- **`load_in_4bit=True`** quantizes the frozen base weights to 4-bit, cutting memory use
  roughly 4×. Under the hood Unsloth uses 4-bit **NF4** quantization with double
  quantization, equivalent to:
  ```python
  BitsAndBytesConfig(load_in_4bit=True,
                     bnb_4bit_quant_type="nf4",
                     bnb_4bit_use_double_quant=True)
  ```
  This is the "Q" in QLoRA.
- **`max_seq_length=2048`** is the maximum context; these prompts are short.
- **`get_chat_template(..., chat_template="qwen3-instruct")`** applies the model's native
  ChatML template so that training and inference use *exactly* the same prompt format.

### 5. Build the training text
Each example becomes a three-message conversation:
system (the task instruction) → user (the HTML) → assistant (the JSON). Each conversation
is rendered to a single string with `tokenizer.apply_chat_template(...)` and stored in a
`text` column. The model is trained to continue this text, i.e. to produce the assistant
JSON given the system + user turns.

### 6. Attach LoRA adapters (the LoRA half of QLoRA)
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
Instead of updating all ~4 billion weights, **LoRA** freezes the 4-bit base model and
trains small low-rank matrices attached to the attention and MLP layers. Only a few
million parameters are trainable, which is what makes fine-tuning fit on a modest GPU (and
produce a tiny adapter file). A 4-bit base model plus LoRA adapters is **QLoRA**.

### 7. Configure the trainer
`SFTTrainer` runs the supervised fine-tuning loop. The important knobs:

| Setting | Value | Why |
|---|---|---|
| `per_device_train_batch_size` | 2 | Fits GPU memory |
| `gradient_accumulation_steps` | 4 | Effective batch size = 8 |
| `num_train_epochs` | 3 | The dataset is small (500 rows) |
| `learning_rate` | 2e-4 | Typical QLoRA learning rate |
| `optim` | `adamw_8bit` | Memory-efficient optimizer |
| `lr_scheduler_type` | `linear` | Warm up, then decay |
| `max_length` | 2048 | Truncation length |

Then `train_on_responses_only(trainer)` masks the prompt tokens so the loss is computed
**only on the assistant's JSON**. Without this, the model would also be trained to
reproduce the user's HTML, wasting capacity.

### 8. Train
`trainer.train()` runs the loop and prints peak VRAM and wall-clock time. This demo
finishes in a few minutes.

### 9. Test the fine-tuned model
The same chat template is applied with `add_generation_prompt=True`, which appends the
assistant header so the model starts writing the JSON. Generation uses Qwen3-Instruct's
recommended sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`).

### 10. Export to GGUF
```python
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
```
**GGUF** is the file format used by Ollama and `llama.cpp`. `q4_k_m` is a good
accuracy-vs-size trade-off (~2.5 GB for a 4B model). The export writes the file to the
`gguf_model_gguf/` folder — copy the `.gguf` out of there when you want to run it locally.

### 11. Run it locally with Ollama
Copy the exported `.gguf` into this repo's folder and make the `Modelfile`'s `FROM`
line point at it, then:
```bash
ollama create qwen3-json-extractor -f Modelfile
ollama run qwen3-json-extractor "Extract the product information:
<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>"
```
The `Modelfile` sets the ChatML template, the sampling defaults and the system prompt so
the local model behaves like the one you trained.

---

## Serving with a Cloudflare tunnel

`serve.sh` builds the model in Ollama and exposes its API on a public
`trycloudflare.com` URL using a
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) —
no Cloudflare account needed.

```bash
./serve.sh
```

It will:

1. Auto-install `ollama`, `cloudflared` and system dependencies like `zstd` if they are
   missing.
2. Verify the `.gguf` referenced by the `Modelfile` is present.
3. Start the Ollama server if it is not already running, then create the model.
4. Open a tunnel (rewriting the `Host` header to a localhost value so Ollama accepts the
   request) and print the public URL.

Once it is running, call the model through the tunnel with Ollama's own API:

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

Override the defaults with environment variables if needed:

```bash
MODEL=my-model PORT=11434 ./serve.sh
```

> **Security:** the quick tunnel has **no authentication** — anyone who has the URL can
> query the model. Ollama itself stays bound to `127.0.0.1` and only the tunnel is public.
> Stop the script with `Ctrl+C` to close it.

---

## Key concepts in one line each

- **Supervised fine-tuning (SFT)** — keep training a pretrained model on input→output
  pairs so it learns a specific task.
- **LoRA** — freeze the base model, train small low-rank adapter matrices instead.
- **QLoRA** — LoRA on top of a 4-bit-quantized base model, to save memory.
- **Chat template** — the exact token format (`<|im_start|>user ...`) the model expects.
  Training and inference must use the same one, or quality collapses.
- **Response-only loss** — mask the prompt so only the answer contributes to the loss.
- **GGUF** — quantized, single-file format for local inference (Ollama / llama.cpp).

---

## Getting started

1. Open `fine-tune.ipynb` in a Jupyter notebook environment.
2. Make sure a CUDA-capable GPU is available — the model is trained in 4-bit to fit modest
   hardware.
3. Place `json_extraction_dataset_500.json` in the working directory, or point step 1 at
   its path.
4. Run the cells in order.

The last cell exports the fine-tuned model to GGUF.

### Requirements
- A CUDA-capable GPU.
- Python with internet access to install the dependencies and download the base model —
  the notebook installs everything it needs.

---

## Customizing the demo

### Try another base model
Swapping the model is a one-line change in step 4. Small-model options:
`unsloth/Qwen3-4B-Instruct-2507` (default), `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`
(smaller/faster), `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`, or
`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`.

### Ideas to extend
- Add a held-out validation split and log JSON validity / exact-match accuracy.
- Constrain decoding with a JSON schema/grammar so output is always valid.
- Increase `num_train_epochs` or LoRA rank `r` if the model underfits.
- Push the adapter or GGUF to the Hugging Face Hub with `push_to_hub_gguf`.
