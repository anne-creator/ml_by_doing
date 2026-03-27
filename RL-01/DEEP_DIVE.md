# RL-01 Deep Dive — Code explained line by line

This document walks through every part of `grpo_colab.py`. Read this while the script runs.

---

## Setup — install dependencies

```python
!pip install trl datasets transformers -q
```

Installs three libraries. `trl` contains the GRPO trainer. `datasets` downloads GSM8K. `transformers` loads the Qwen model. The `!` prefix tells Colab to run this as a terminal command. `-q` means quiet, less output noise.

---

## Part 1 — Dataset

```python
dataset = load_dataset("openai/gsm8k", "main", split="train")
```

Downloads the GSM8K dataset from HuggingFace. `"main"` is the standard version. `split="train"` gives us the training portion, 7473 problems total. Each row has two fields: `question` and `answer`.

```python
dataset = dataset.select(range(500))
```

Keeps only the first 500 problems. We do not need all 7473 for this experiment. Fewer rows means faster training.

```python
def format_prompt(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"Solve this math problem. At the end write your final answer as a plain number after '####'.\n\nProblem: {example['question']}"
            }
        ]
    }
```

Wraps each question into a chat message format. The `role: "user"` tells the model this is a human asking a question, the same format used when you chat with Claude or ChatGPT. Without this wrapping, the model does not know it is being asked a question.

```python
dataset = dataset.map(format_prompt)
```

Runs `format_prompt` on all 500 rows. Adds a new `prompt` column to each row. The original `question` and `answer` columns stay untouched. The trainer reads `prompt` to feed the model, and reads `answer` to pass into the reward function.

---

## Part 2 — Reward function

```python
def extract_answer(text):
    match = re.search(r'####\s*(-?\d+(?:,\d+)?(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None
```

A helper function that looks for `####` followed by a number in any text. The regex handles negatives like `-5`, decimals like `3.14`, and comma formatting like `1,000`. Returns the number as a string, or `None` if not found. This is why you see `Predicted: None` when the model does not write `####`.

```python
def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer", [""] * len(completions))
    rewards = []
```

`trl` calls this function automatically after every generation. `completions` is the list of answers the model just wrote. `kwargs` contains extra fields from the dataset including the correct `answer` column.

```python
    if predicted and correct and predicted == correct:
        reward = 1.0
    elif predicted is not None:
        reward = 0.3
    else:
        reward = 0.0
```

Three-level reward. `1.0` for a correct answer. `0.3` for right format but wrong number, this partial credit helps the model learn the `####` format faster because it gets some signal even before it gets the math right. `0.0` for no `####` at all.

```python
    return rewards
```

Returns the list of scores to `trl`. GRPO uses these to compute the group advantage: each score is compared to the average of the group, and weights are nudged accordingly.

---

## Part 3 — Config

```python
training_args = GRPOConfig(
    output_dir="./grpo-output",
```

Folder where model checkpoints are saved after training.

```python
    max_steps=200,
```

Run for 200 weight updates. With batch size 4, this means 800 problems seen total. Enough to see reward climb.

```python
    per_device_train_batch_size=4,
```

Take 4 problems per step. The T4 GPU has 16GB dedicated VRAM so it handles this comfortably. On a MacBook Air with shared 16GB you would drop this to 2.

```python
    num_generations=4,
```

Generate 4 answers per problem. So 4 problems × 4 answers = 16 total scored per step. More generations means more contrast for GRPO to compare within the group, which means a stronger learning signal.

```python
    max_completion_length=256,
```

The model can write up to 256 tokens per answer. More room than the Mac version (128) because the T4 has headroom for it. This lets the model show its reasoning before writing `####`.

```python
    bf16=False,
    fp16=True,
```

Precision settings. `bf16` is not supported on the free T4 GPU. `fp16` is, and it cuts memory usage roughly in half compared to the default `float32`. Always set both when running on Colab free tier.

---

## Part 4 — Trainer

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
```

The model we are fine-tuning. `Qwen` is the company (Alibaba). `2.5` is the version. `0.5B` means 500 million parameters. `Instruct` means it has already been fine-tuned for chat. `trl` downloads it automatically from HuggingFace on first run, about 1GB.

```python
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_fn,
```

Wires the four pieces together. Config from Part 3, dataset from Part 1, reward function from Part 2. `trl` handles everything else: forward pass, calling the reward function, running GRPO math, backpropagation, weight update.

```python
trainer.train()
```

Starts the loop. Runs for 200 steps. Does not return until done.

---

## Part 5 — Test your model

```python
model = AutoModelForCausalLM.from_pretrained("./grpo-output")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

Loads your trained model back from the checkpoint folder. The tokenizer converts text to numbers the model understands and back again.

```python
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Runs one question through your trained model. `return_tensors="pt"` means return PyTorch tensors. `max_new_tokens=128` caps the response length. `decode` converts the output tensor back to readable text.

---

## Part 6 — Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copytree("./grpo-output", "/content/drive/MyDrive/grpo-output")
```

Colab's filesystem is temporary. When your session disconnects, everything in `./grpo-output` is gone. This copies your trained model to Google Drive before that happens. Run this as a separate cell after training finishes.

---

## What the training output means

```
[Sample 0]
  Output    : #### 72
  Predicted : 72
  Correct   : 30
  Reward    : 0.0
```

`Output` is what the model wrote. `Predicted` is what the reward function extracted after `####`. `Correct` is the dataset answer. `Reward` is the score. This sample got the format right (0.3 reward) but the number wrong.

```
Step  Training Loss
10    0.131
20    0.000
30    0.087
```

`Training Loss` is how much the model changed that step. `0.000` means all answers in the group scored the same, usually all `0.0`, so GRPO had nothing to compare and made no update. Non-zero means at least one answer scored differently, GRPO had a real signal, and the weights moved.

---

## Key concepts used in this experiment

| Term | What it means here |
|---|---|
| Weights / Parameters | The 500 million numbers inside Qwen. Training nudges these. |
| Step | One weight update. 200 steps = 200 updates. |
| Batch | 4 problems per step. |
| Epoch | One full pass through all 500 problems. 200 steps = 1.6 epochs. |
| Reward | Score from your reward function. 1.0, 0.3, or 0.0. |
| Loss | How much the model changed this step. 0 = nothing learned. |
| Token | A chunk of text. `max_completion_length=256` means 256 tokens max. |
| Checkpoint | Saved model weights at a point in training. Stored in `./grpo-output`. |
| Forward pass | Model reads the question and generates an answer. |
| Backpropagation | PyTorch traces which weights contributed to the answer and computes how much to adjust each one. |
