# 📊 TOPSIS-Based Evaluation of Sentence Similarity Models

## 📌 Project Overview

This project evaluates multiple Hugging Face pre-trained sentence embedding models using the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method.

The objective is to rank sentence similarity models based on both semantic performance and computational efficiency.

---

# 🎯 Problem Statement

Select the best sentence similarity model using multi-criteria decision-making.

We compare models based on:

- Semantic Textual Similarity (STS Score)
- Inference Time
- Model Size
- Embedding Dimension
- Cosine Similarity

---

# 🧠 Sentence Pairs Used for Testing

```python
sentence_pairs = [
    ("Machine learning is a subset of artificial intelligence.",
     "Artificial intelligence includes machine learning as one of its parts."),

    ("Deep learning models require large amounts of data.",
     "Neural networks perform better when trained with more data."),

    ("The stock market fluctuates daily.",
     "Neural networks are used in computer vision.")
]
