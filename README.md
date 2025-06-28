## 🤖 BERT-based Question Answering System (SQuAD v2.0)

This project implements a **Question Answering (QA)** system powered by **BERT (Bidirectional Encoder Representations from Transformers)**, fine-tuned on the **SQuAD v2.0 dataset**. The system can answer questions based on a given context and identify when a question is unanswerable, making it robust and practical for real-world use.

---

## 🌟 Project Highlights

- **Dataset**: Processes and trains on the SQuAD v2.0 dataset (100,000+ question-answer pairs).
- **Model**: Fine-tunes a pre-trained BERT model for accurate question answering.
- **Evaluation**: Measures performance with F1 score and Exact Match (EM) metrics.
- **Custom QA**: Allows users to input custom contexts and questions for inference.

---

## 🎯 Example in Action

**Context**:  
_The Eiffel Tower, a global cultural icon of France, is located in Paris._

**Question**:  
_Where is the Eiffel Tower located?_

**Answer**:  
_Paris_

**Unanswerable Question**:  
_What is the Eiffel Tower's favorite color?_

**Answer**:  
_No answer_

---

## 📂 Project Structure

| File/Folder                | Description                                                |
|---------------------------|------------------------------------------------------------|
| `QA_BERT_Updated.ipynb`   | Jupyter notebook with code for training, evaluation, and inference |
| `evaluate-v2.0.py`        | Official SQuAD v2.0 evaluation script                      |
| `predictions.json`        | Stores model predictions for the validation set            |
| `bert-qa-checkpoints/`    | Directory for saved model checkpoints (not included in repo)|
| `train-v2.0.json`         | SQuAD v2.0 training dataset (download separately)          |
| `dev-v2.0.json`           | SQuAD v2.0 validation dataset (download separately)        |
| `README.md`               | This documentation file                                   |
| `.gitignore`              | Excludes large files (e.g., checkpoints, datasets)         |

---

## 🧠 How It Works (Beginner-Friendly)

1. **Tokenization**: Text is converted into numerical tokens that BERT understands.
2. **Training**: The BERT model is fine-tuned on SQuAD v2.0 question-answer pairs.
3. **Prediction**: The model identifies the most relevant text span in the context as the answer or flags the question as unanswerable.
4. **Evaluation**:
   - **Exact Match (EM)**: Percentage of answers that match the ground truth exactly.
   - **F1 Score**: Measures overlap between predicted and true answers.

---

## 🚀 Getting Started

### 🖥️ Option 1: Run in Google Colab (Recommended for Beginners)

1. Open `QA_BERT_Updated.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Download `train-v2.0.json` and `dev-v2.0.json` from the SQuAD website and upload them to Colab.
3. Run all cells in sequence. ✅ Make sure to enable GPU runtime for faster training.

### 🐍 Option 2: Run Locally (Advanced Users)

```bash
# Clone the repository
git clone https://github.com/yourusername/bert-question-answering-squad2.git
cd bert-question-answering-squad2

# (Optional) Set up a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset files
# train-v2.0.json and dev-v2.0.json (not included in repo)

# Launch the notebook
jupyter notebook QA_BERT_Updated.ipynb
````

---

## 📋 Requirements

This project requires the following Python libraries:

```text
transformers==4.31.0
datasets==2.14.0
evaluate==0.4.0
torch==2.0.1
scikit-learn==1.3.0
tqdm==4.65.0
```

Install them with:

```bash
pip install transformers==4.31.0 datasets==2.14.0 evaluate==0.4.0 torch==2.0.1 scikit-learn==1.3.0 tqdm==4.65.0
```

✅ Make sure you’re using **Python 3.8+** and have a **compatible GPU** if training.

---

## 📊 Evaluating the Model

You can evaluate your model on the validation set using:

```bash
python evaluate-v2.0.py dev-v2.0.json predictions.json
```

### Key Metrics:

* `exact`: Percentage of exact matches between predicted and true answers.
* `f1`: F1 score based on word overlap between predicted and true answers.
* `HasAns_*`: Performance on answerable questions.
* `NoAns_*`: Performance on unanswerable questions.

---

## 🔍 About SQuAD v2.0

The **Stanford Question Answering Dataset (SQuAD)** v2.0 is a benchmark QA dataset with:

* 100,000+ answerable questions based on Wikipedia
* 50,000+ unanswerable questions that test a model’s ability to say “no answer”

🔗 [Official SQuAD Website](https://rajpurkar.github.io/SQuAD-explorer/)

---

## 🔧 Enhancements to Explore

* 🚀 **Deploy the Model** using Gradio or Streamlit
* 🧠 **Upgrade** to `bert-large` or `roberta-base` for better performance
* 🖼️ **Build a GUI** that lets users input custom text and ask questions
* ⚡ **Optimize** using mixed-precision or quantization for faster inference

---

## 🙌 Acknowledgments

* Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
* Dataset: [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)
* Special thanks to the open-source community

---
