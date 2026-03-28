> You do not need to read the tutorial PDF first. This handbook is standalone. Read it before, during, or after running the script.

Still has questions while reading? Toss the handbook into Claude or Gemini, then, start ask questions. 

Remember, Machine Learning knowledge can not be understand all at once, but it does not prevent you from using and training them, keep going while try to grasp every concepts!
> 

Alterntive to read this notion page inside, same content, better readibility! 

---

# Part 1 — What you are about to do

You are going to train an AI model in your browser, for free, in about 25 minutes.

Not simulate it. Not watch a video of it. Actually run it.

The algorithm is called **GRPO**. It is the same algorithm DeepSeek used to train DeepSeek R1, the model that matched GPT-4 at a fraction of the cost. You are running a tiny version of that exact loop.

The model starts not knowing what output format to use. By the end, it will have learned on its own. You will watch that happen line by line in your terminal.

That is the whole experiment.

---

# Part 2 — Three things you need to know before running

**What agentic RL is**

A way to train AI models by scoring their answers instead of hand-labeling them. You define what good looks like. The model figures out how to get there on its own.

**What GRPO is**

One specific agentic RL algorithm. Invented by DeepSeek. The key idea: ask the model the same question multiple times, compare the answers to each other, reward the ones that did better than average. No second model needed. No human labels needed.

**What training a model means**

The model you are using, Qwen2.5-0.5B, already exists. It was pre-trained on hundreds of millions of internet pages. It can already talk and answer questions. **Training here means taking that existing model and making it better at one specific job**: solving math problems and writing the answer after `####`.

---

# Part 3 — Run it

Wait for it to finish. Every time you start a new Colab session, you need to run this again.

**Step 1.** Go to [colab.research.google.com](http://colab.research.google.com) → new notebook → Runtime → Change runtime type → T4 GPU → Save.

**Before anything else, install dependencies.** Paste this into the first Colab cell and run it with `Shift+Enter`:

```python
!pip install trl datasets transformers -q
```

**Step 2.** Run the install cell above.

**Step 3.** Create a new cell. Paste and run the script here https://github.com/anne-creator/ml_by_doing/blob/main/RL-01/main_script.py

**Step 4.** Grab a coffee. Takes about 25 minutes.

While it runs, keep reading.

---

# Part 4 — What is actually happening

## The data

The script uses a dataset called GSM8K — 500 grade school math problems from OpenAI. Each row looks like this:

> **Question:** Natalia sold clips to 48 of her friends in April, then sold half as many in May. How many clips did she sell altogether?
> 

> **Correct answer:** #### 72
> 

The `####` is a format convention we defined. We told the model: write your final answer after `####`. The reward function checks if the number after `####` matches the correct answer.

Source: [HuggingFace GSM8K](https://huggingface.co/datasets/openai/gsm8k) — 7,473 training rows total. The script uses 500.

## What one step actually looks like

A **step** is the smallest unit of training. One step = one weight update. Here is exactly what happens in one step:

**1. Take 4 problems from the dataset**

```
Problem A: Natalia sold clips to 48 friends in April, half as many in May. Total?
Problem B: Weng earns $12/hr babysitting. She did 50 minutes yesterday. How much?
Problem C: Betty saving for a $100 wallet. She has $15 + half from grandma...
Problem D: Julie baked 3 dozen cookies. Ate 2, gave sister 1. How many remain?
```

**2. Generate 4 answers per problem** = 16 total answers this step

```
Problem A, Answer 1: "...#### 72"   → predicted=72, correct=72  → reward 1.0
Problem A, Answer 2: "...#### 72"   → predicted=72, correct=72  → reward 1.0
Problem A, Answer 3: "...#### 96"   → predicted=96, correct=72  → reward 0.3
Problem A, Answer 4: "...#### 48"   → predicted=48, correct=72  → reward 0.3
```

**3. Compare scores to the group average**

```
Group average = (1.0 + 1.0 + 0.3 + 0.3) / 4 = 0.65

Answer 1: 1.0 - 0.65 = +0.35  → nudge weights UP
Answer 2: 1.0 - 0.65 = +0.35  → nudge weights UP
Answer 3: 0.3 - 0.65 = -0.35  → nudge weights DOWN
Answer 4: 0.3 - 0.65 = -0.35  → nudge weights DOWN
```

This is the "Group Relative" part of GRPO. The model is not told what the right answer is. It only learns what was relatively better or worse within the group.

**4. Update the weights.** Done. One step complete. Move to next 4 problems. Repeat 200 times.

## What you actually see printed

**Step 1 — model has no format yet:**

```
[Sample 0]
  Output    : To solve this problem, we need to determine the percentage...
  Predicted : None
  Correct   : 30
  Reward    : 0.0
```

**Around step 7 — format appears:**

```
[Sample 0]
  Output    : #### 72
  Predicted : 72
  Correct   : 30
  Reward    : 0.3
```

Number wrong. Format learned. Nobody told it to write `####`. It discovered on its own that outputs with `####` occasionally matched the correct answer and got rewarded.

**By step 50 — format consistent:**

```
[Sample 0]
  Output    : #### 30
  Predicted : 30
  Correct   : 30
  Reward    : 1.0
```

That behavior shift is real learning from 80 lines of code.

---

# Part 5 — Understanding weights

**What weights actually are**

The model you downloaded, `Qwen2.5-0.5B`, is a single file of about 1GB. Inside that file are 500 million **floating point numbers**. 

Those numbers encode everything the model knows: grammar, math reasoning, how to follow instructions. The 0.5B in the name means 500 million. That is the parameter count.

**How the model uses those numbers**

When the model reads a question, the text is first converted into numbers (called tokens). Those numbers flow through 28 layers one by one. Each layer does a matrix multiplication using its share of the 500 million weights. The output of the final layer is a probability score over every possible next word. The model **picks the highest probability word, adds it to the output**, and runs the whole thing again to pick the next word.

That is how it generates text. One token at a time. 28 layers of math each time.

**How training changes those numbers**

When GRPO computes that Answer 1 scored above average and Answer 3 scored below average, it needs to figure out which of those 500 million weights contributed to each answer and by how much.

PyTorch does this automatically using a process called **backpropagation**. During generation, PyTorch kept a record of every multiplication and addition that happened. After scoring, it walks backward through that receipt and computes a sensitivity score for each weight. That score is called a gradient.

High gradient = this weight had a big influence on the output. Near-zero gradient = this weight barely participated.

Weights with high gradients that contributed to above-average answers get nudged up slightly. Weights that contributed to below-average answers get nudged down. After 200 steps, thousands of tiny nudges have accumulated into a model that behaves measurably differently than before.

**The one line summary:** **Training** = nudging 500 million numbers in the right direction, over and over, guided by a score.

---

# Part 6 — How to read your results

After training finishes, you will see a **loss table** and a stream of sample outputs. Here is how to diagnose what you saw.

**Reward stayed at 0 the whole time**

This means every pair of answers in every group scored the same, usually both 0.0. GRPO had no contrast to learn from. The model did not update. This happened in the early Mac version of this experiment because 50 steps was not enough and the binary reward gave no signal. The Colab version uses partial credit (0.3 for right format) to fix this.

**Loss is 0 for most steps**

Same cause. No contrast within the group = nothing to update = loss 0. A non-zero loss step is a good sign, it means the model had something to learn that step.

**Predicted went from None to a number**

Good sign. The model learned the `####` format. This is the main behavioral change you are looking for in 200 steps.

**Reward is still mostly 0.3 not 1.0 at step 200**

Expected. The model learned the format but not the math. Getting the math right requires more steps (500+) and a larger model (1.5B+).

**The experiment worked if:** Predicted stopped being None somewhere around step 7 to 15. Everything else is a bonus.

---

# Part 7 — What next

Three paths depending on what you want:

**Path 1: See a real before/after result**

Swap the model to `Qwen2.5-1.5B-Instruct`, increase `max_steps` to 500, add a baseline test before training. Run overnight on Colab. You will see reward actually climb.

**Path 2: Understand the math behind this**

Watch Andrej Karpathy's "Neural Networks: Zero to Hero" on YouTube. Start from video 1. Everything you ran today will make deeper sense after the first two videos.

**Path 3: Apply this to a real task**

Replace GSM8K with your own dataset. Replace the reward function with something that scores what you care about. The loop is identical. GRPO does not care what the task is.

---

# Part 8 — Glossary

It is okay to not fully understand these yet. They will click eventually.

**Weights / Parameters**

The 500 million numbers inside the model. They encode everything it knows. Training nudges these numbers in the right direction. Same thing, two names.

**Step**

One weight update. Takes a batch of questions, generates answers, scores them, nudges weights. 200 steps = 200 updates.

**Batch**

How many problems the model sees per step. This experiment uses 4.

**Epoch**

One full pass through the entire dataset. 200 steps on 500 problems with batch size 4 = ~1.6 epochs.

**Token**

A chunk of text, usually a word or part of a word. `max_completion_length=256` means the model writes at most 256 tokens per answer.

**Reward**

The score your reward function gives each answer. This experiment uses 1.0 for correct, 0.3 for right format wrong number, 0.0 for no format at all.

**Loss**

How much the model changed this step. Loss = 0 means nothing learned. Non-zero = real update happened.

**Gradient**

The sensitivity score for each weight. Tells PyTorch how much to adjust each weight based on the reward signal.

**Backpropagation**

PyTorch's mechanism to compute gradients. Walks backward through the receipt of calculations the model made during generation. You never write this yourself.

**Forward pass**

The model reading a question and generating an answer. Data flows forward through all 28 layers.

**GPU**

The chip that runs the math. Much faster than CPU for ML because it does millions of calculations in parallel. T4 is the free one Colab gives you.

**Checkpoint**

A saved snapshot of the model weights at a point in training. `./grpo-output` is your checkpoint folder. Think of it as a Git commit for model weights.

**Adapter vs full model**

GRPO saves only what changed, not the full model. Like a Git diff. To get a portable standalone model, merge the base model and adapter using `merge_and_unload()`. The merged version is saved in `./grpo-final`.

**Fine-tuning**

Training a model that already exists to be better at a specific task. What this experiment does. The opposite of training from scratch.

---

# Sources

- Dataset: [HuggingFace GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- Model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- GRPO paper: [DeepSeekMath arxiv](https://arxiv.org/abs/2402.03300)
- trl library: [github.com/huggingface/trl](http://github.com/huggingface/trl)
- Full annotated script: https://github.com/anne-creator/ml_by_doing/blob/main/RL-01/main_script.py

---

Built by Anne · [LinkedIn](https://www.linkedin.com/in/anneliu49/) · [ml-by-doing](https://github.com/anne-creator/ml_by_doing)

⭐ If this saved you a Google search, a star keeps the project going.
