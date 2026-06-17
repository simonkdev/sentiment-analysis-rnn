# Sentiment Analysis with RNN

A **Recurrent Neural Network** implementation for **sentiment analysis** of IMDB movie reviews. This project classifies text as positive or negative using a custom RNN implementation.

---

## 📌 About

### Purpose
This project implements sentiment analysis on the **IMDB movie review dataset** using a Recurrent Neural Network (RNN) with the following characteristics:
- **Dataset**: IMDB reviews (positive/negative labels)
- **Task**: Binary text classification
- **Implementation**: Custom RNN from scratch using NumPy

---
## ✨ Features

- **Binary Classification**: Positive or negative sentiment prediction
- **Custom RNN Implementation**: Built from scratch without deep learning frameworks
- **Token Embedding**: Learned 50-dimensional word embeddings
- **Training Pipeline**: Full training and evaluation workflow
- **Web Interface**: Interactive sentiment analysis
- **API Access**: Programmatic classification

---
## 🏗️ Model Architecture

| Component | Details |
|-----------|---------|
| **Input Layer** | Token embeddings (50 dimensions) |
| **Hidden Layer** | 8 LSTM-like neurons with tanh activation |
| **Output Layer** | 2 neurons (positive/negative) with softmax activation |
| **Sequence Length** | 150 tokens (padded/truncated) |

---
## 🧠 Token Embedding Technique

### Embedding Layer
- **Embedding Dimension**: 50
- **Vocabulary Size**: Dynamic (built from training data)
- **Initialization**: Random normal (mean=0, std=0.01)
- **Update Method**: Gradient descent with clipping (range: [-1.0, 1.0])
- **Learning Rate**: 0.0006

### Tokenizer
- **Type**: Word-level tokenizer
- **Preprocessing**:
  - Lowercasing (enabled by default)
  - Filtering special characters (`!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n`)
  - Splitting on whitespace
- **Sequence Handling**:
  - Maximum sequence length: 150 tokens
  - Padding: Pre-padding (zeros)
  - Truncation: Pre-truncation
- **Out-of-Vocabulary**: Tokens not in vocabulary are skipped

### Training Details
- **Optimizer**: Custom BPTT with gradient clipping
- **Learning Rate**: 0.0006
- **Batch Size**: 256
- **Loss Function**: Categorical cross-entropy
- **Gradient Clipping**: [-1.0, 1.0] for all parameters
- **Shuffling**: Enabled during training

---
## 🚀 Usage

### Prerequisites
- Python 3.11+
- Devenv (recommended)

### Setup

```sh
git clone https://github.com/simonkdev/sentiment-analysis-rnn
cd sentiment-analysis-rnn
devenv shell
```

### Running the Web Interface

```sh
python web_page.py
```
The app will be available at `http://localhost:5000`.

### Training the Model
```sh
python train.py
```
This will:
1. Load the IMDB dataset from `data/`
2. Tokenize text and build vocabulary
3. Train the RNN model with 8 hidden neurons
4. Save model parameters to `params/`

---
## 📁 Project Structure

```
sentiment-analysis-rnn/
├── main_rnn.py              # Main RNN implementation
├── train.py                 # Training script
├── server.py                # Web server
├── api.py                   # API endpoints
├── web_page.py              # Frontend web page
├── src/
│   ├── activation.py        # Activation functions (tanh, softmax)
│   ├── backprop.py          # Backpropagation logic
│   ├── data_prep.py         # Data preprocessing and tokenization
│   ├── forward_pass.py      # Forward pass implementation
│   └── initialization.py     # Weight initialization
├── data/                    # IMDB dataset storage
├── params/                  # Model parameters (weights, tokenizer, embeddings)
├── requirements.txt          # Python dependencies
├── devenv.nix               # Devenv packages
├── devenv.yaml              # Devenv config
└── devenv.lock              # Locked dependencies
```

### File Descriptions

| File | Purpose |
|------|---------|
| `main_rnn.py` | Core RNN model implementation |
| `train.py` | Training pipeline |
| `server.py` | Web server |
| `api.py` | API endpoint definitions |
| `web_page.py` | Frontend interface |
| `src/activation.py` | Activation functions (tanh, softmax) |
| `src/backprop.py` | Backpropagation through time |
| `src/data_prep.py` | Data preprocessing, tokenization, and embedding |
| `src/forward_pass.py` | Forward pass for RNN |
| `src/initialization.py` | Weight initialization (Xavier/Glorot) |
| `data/` | IMDB dataset |
| `params/` | Saved model weights, tokenizer, and embedding matrix |

---
## 📜 License

This project is licensed under the **MIT License**.

Copyright (c) 2026 Simon Korten
```

---
**Next repository?** Just name it!
