# 🤖 ML by Doing

> **Learn ML by actually running it.**
> Hands-on experiments for developers with no ML background — copy, paste, run, understand.

---

## 📖 About This Repo

This repository is a collection of **self-contained mini-projects**, each living in its own folder. Every lesson is designed to be:

- **Minimal** — a single concept per experiment
- **Runnable** — working code you can execute right away
- **Explainable** — comments and a `README.md` in each folder walking you through the *why*

No prior machine learning experience required. If you can write a `for` loop, you can follow along.

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.9 | [python.org](https://www.python.org/downloads/) |
| pip | latest | bundled with Python |
| Jupyter (optional) | latest | `pip install notebook` |

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/anne-creator/ml_by_doing.git
cd ml_by_doing

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Navigate to any lesson folder and follow its README
cd 01_linear_regression
pip install -r requirements.txt   # if present
python main.py
```

---

## 📚 Lessons

Each folder is an independent lesson. Click a lesson to jump to it once it's added to the repo.

| # | Folder | Topic | Difficulty |
|---|--------|-------|------------|
| 01 | [Linear Regression](./01_linear_regression/) | Predict a continuous value from features | 🟢 Beginner |
| 02 | [Logistic Regression](./02_logistic_regression/) | Binary classification from scratch | 🟢 Beginner |
| 03 | [K-Nearest Neighbors](./03_k_nearest_neighbors/) | Classify by proximity | 🟢 Beginner |
| 04 | [Decision Trees](./04_decision_trees/) | Rule-based splitting for classification & regression | 🟡 Intermediate |
| 05 | [Random Forests](./05_random_forests/) | Ensembles of decision trees | 🟡 Intermediate |
| 06 | [Naive Bayes](./06_naive_bayes/) | Probabilistic text classification | 🟡 Intermediate |
| 07 | [Support Vector Machines](./07_support_vector_machines/) | Maximum-margin classifiers | 🟡 Intermediate |
| 08 | [K-Means Clustering](./08_k_means_clustering/) | Unsupervised grouping of data | 🟡 Intermediate |
| 09 | [Principal Component Analysis](./09_pca/) | Dimensionality reduction & visualization | 🟡 Intermediate |
| 10 | [Neural Network from Scratch](./10_neural_network_scratch/) | Build a feedforward net with NumPy only | 🔴 Advanced |
| 11 | [Convolutional Neural Networks](./11_cnn/) | Image recognition with CNNs | 🔴 Advanced |
| 12 | [Recurrent Neural Networks](./12_rnn/) | Sequence modeling & time series | 🔴 Advanced |
| 13 | [Transfer Learning](./13_transfer_learning/) | Fine-tune a pretrained model | 🔴 Advanced |
| 14 | [Reinforcement Learning Intro](./14_reinforcement_learning/) | Agents, rewards, and Q-learning basics | 🔴 Advanced |

> 🗓️ **New lessons are added regularly.** Watch/Star the repo to get notified.

---

## 🗂️ Folder Structure

Every lesson folder follows the same layout:

```
XX_lesson_name/
├── README.md          # Concept explanation + step-by-step guide
├── main.py            # Runnable entry point
├── notebook.ipynb     # (optional) Jupyter notebook version
├── requirements.txt   # (optional) Python dependencies
└── data/              # (optional) Sample datasets
```

---

## 🤝 Contributing

Contributions, suggestions, and new lesson ideas are welcome!

1. Fork this repository
2. Create a new branch: `git checkout -b lesson/your-topic`
3. Add your lesson folder following the structure above
4. Open a Pull Request with a short description of the experiment

Please keep each lesson **self-contained** and **beginner-friendly**.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

*Happy experimenting! 🧪*
