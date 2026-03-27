import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# ── PART 1: DATASET ───────────────────────────────────────
dataset = load_dataset("openai/gsm8k", "main", split="train")  # download GSM8K, 7473 math problems
dataset = dataset.select(range(500))                            # use 500 problems for more training signal

def format_prompt(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"Solve this math problem. At the end write your final answer as a plain number after '####'.\n\nProblem: {example['question']}"
            }
        ]
    }

dataset = dataset.map(format_prompt)                           # adds 'prompt' column to all 500 rows

# ── PART 2: REWARD FUNCTION ───────────────────────────────
def extract_answer(text):
    match = re.search(r'####\s*(-?\d+(?:,\d+)?(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer", [""] * len(completions))
    rewards = []

    for i, completion in enumerate(completions):
        generated = completion[0]["content"] if isinstance(completion, list) else completion
        predicted = extract_answer(generated)
        correct = extract_answer(answers[i])

        if predicted and correct and predicted == correct:
            reward = 1.0                                       # correct answer
        elif predicted is not None:
            reward = 0.3                                       # right format, wrong number
        else:
            reward = 0.0                                       # no #### found

        rewards.append(reward)

        if i < 2:
            print(f"\n[Sample {i}]")
            print(f"  Output    : {generated[:120]}...")
            print(f"  Predicted : {predicted}")
            print(f"  Correct   : {correct}")
            print(f"  Reward    : {reward}")

    return rewards

# ── PART 3: CONFIG ────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="./grpo-output",
    max_steps=200,                        # 200 steps to see reward actually climb
    per_device_train_batch_size=4,        # T4 has 16GB dedicated VRAM, handles 4
    num_generations=4,                    # 4 answers per problem, more contrast for GRPO
    max_completion_length=256,            # room to reason before writing ####
    logging_steps=10,
    report_to="none",
    bf16=False,                           # T4 free tier does not support bfloat16
    fp16=True,                            # float16 cuts memory in half, T4 handles this
)

# ── PART 4: TRAIN ─────────────────────────────────────────
print("\nLoading model and starting training...\n")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_fn,
)

trainer.train()
print("\nDone. Check the reward numbers above.")

# ── PART 5: TEST ──────────────────────────────────────────
# ── MERGE AND SAVE FULL MODEL ─────────────────────────────
# This creates a complete standalone model you can load anywhere
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

!pip install peft -q

# load base model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# apply your trained weights on top
model = PeftModel.from_pretrained(base, "./grpo-output")

# merge into one single model and save it
model = model.merge_and_unload()
model.save_pretrained("./grpo-final")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.save_pretrained("./grpo-final")

print("Full model saved to ./grpo-final")

# ── PART 6: SAVE TO GOOGLE DRIVE ──────────────────────────
# Uncomment and run separately before your session ends
#
# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# shutil.copytree("./grpo-output", "/content/drive/MyDrive/grpo-output")
