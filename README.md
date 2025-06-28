
````markdown
# 🤖 BERT-based Question Answering System (SQuAD v2.0)

This project builds a **Question Answering (QA)** system using **BERT (Bidirectional Encoder Representations from Transformers)**. It is trained on the **SQuAD v2.0 dataset**, allowing it to answer questions from a given context and also know when **no answer is possible**.

---

## 📍 What It Does

- Loads and processes the **SQuAD v2.0** dataset (100k+ Q&A pairs)
- Fine-tunes a pre-trained **BERT model** on this dataset
- Evaluates the model using **F1 score and Exact Match**
- Supports **custom question answering** using a trained model

---

## 🧪 Example Demo

**Context**:  
> The Eiffel Tower is located in Paris.

**Question**:  
> Where is the Eiffel Tower?

**Answer**:  
> Paris

---

## 📁 Project Structure

| File / Folder | Description |
|---------------|-------------|
| `QA_BERT_Updated.ipynb` | Full notebook with training, evaluation, and inference |
| `evaluate-v2.0.py` | Official evaluation script for SQuAD v2.0 |
| `predictions.json` | Model predictions on validation set |
| `bert-qa-checkpoints/` | Saved model checkpoints (not uploaded to GitHub) |
| `train-v2.0.json` / `dev-v2.0.json` | Dataset files (SQuAD v2.0 — not uploaded) |
| `README.md` | This documentation file |
| `.gitignore` | Prevents large files (checkpoints, datasets) from being tracked |

---

## 💡 How It Works (Beginner Friendly)

1. **Tokenization**: Converts text into numbers the model can understand
2. **Model Training**: Fine-tunes BERT on Q&A pairs from Wikipedia
3. **Prediction**: Extracts the best span of text as the answer
4. **Evaluation**: Compares predictions with true answers using:
   - **Exact Match (EM)** – how many answers are exactly right
   - **F1 Score** – how close the predicted answers are in terms of overlap

---

## 🚀 How to Run It

### 🖥️ Option 1: Run in Google Colab

1. Open `QA_BERT_Updated.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `train-v2.0.json` and `dev-v2.0.json`
3. Run all cells sequentially

### 🐍 Option 2: Run Locally (Advanced Users)

```bash
git clone https://github.com/yourusername/bert-question-answering-squad2.git
cd bert-question-answering-squad2

# Set up environment
pip install -r requirements.txt

# Train and test using Jupyter or Python
jupyter notebook QA_BERT_Updated.ipynb
````

---

## 🧾 Requirements

```text
transformers
datasets
evaluate
torch
scikit-learn
tqdm
```

To install:

```bash
pip install transformers datasets evaluate torch scikit-learn tqdm
```

---

## 📊 Model Evaluation

You can evaluate model performance using the included `evaluate-v2.0.py` script:

```bash
python evaluate-v2.0.py dev-v2.0.json predictions.json
```

**Outputs**:

* `exact`: Exact match score
* `f1`: F1 overlap score
* `HasAns_`, `NoAns_`: Performance split by answerable/unanswerable questions

---

## 🧠 What is SQuAD v2.0?

SQuAD (Stanford Question Answering Dataset) v2.0 contains:

* 100,000+ questions with answers
* 50,000+ **unanswerable** questions

This forces models not just to **extract correct spans**, but also to **know when to say "no answer"**.

[📎 Official Dataset Site](https://rajpurkar.github.io/SQuAD-explorer/)

---

## ✨ Features You Can Add Next

* Deploy using **Gradio** or **Streamlit**
* Use a bigger model like `bert-large` or `roberta-base`
* Build a UI that allows users to paste text and ask questions

---

## 🙌 Credits

* Built using [Hugging Face Transformers](https://huggingface.co/transformers/)
* Dataset: [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)
* Special thanks to the open-source community

---
