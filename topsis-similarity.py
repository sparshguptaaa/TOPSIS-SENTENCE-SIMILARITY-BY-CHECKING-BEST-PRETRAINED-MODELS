import numpy as np
import pandas as pd
import torch
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



models = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniLM-L12-v2": "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "MPNet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "E5-base-v2": "intfloat/e5-base-v2",
    "BGE-base-en-v1.5": "BAAI/bge-base-en-v1.5"
}


sentences = [
    "The Taj Mahal is located in India.",
    "Quantum computing uses qubits instead of classical bits."
]




sts_scores = {
    "MiniLM-L6-v2": 0.85,
    "MiniLM-L12-v2": 0.87,
    "MPNet-base-v2": 0.90,
    "E5-base-v2": 0.89,
    "BGE-base-en-v1.5": 0.91
}

results = []


for name, model_path in models.items():
    print(f"\nLoading {name}...")
    model = SentenceTransformer(model_path)

    # Model size (MB)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)

    # Embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()

    # Inference time
    start_time = time.time()
    embeddings = model.encode(sentences)
    inference_time = time.time() - start_time

    # Cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    results.append([
        name,
        sts_scores[name],
        inference_time,
        model_size,
        embedding_dim,
        similarity
    ])

columns = [
    "Model",
    "STS Score",
    "Inference Time",
    "Model Size (MB)",
    "Embedding Dimension",
    "Cosine Similarity"
]

df = pd.DataFrame(results, columns=columns)
print("\nDecision Matrix:")
print(df)


weights = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
benefit = np.array([1, 0, 0, 0, 1])

matrix = df.iloc[:, 1:].values.astype(float)


norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))


weighted_matrix = norm_matrix * weights


ideal_best = np.max(weighted_matrix * benefit, axis=0) + \
             np.min(weighted_matrix * (1 - benefit), axis=0)

ideal_worst = np.min(weighted_matrix * benefit, axis=0) + \
              np.max(weighted_matrix * (1 - benefit), axis=0)


dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))


scores = dist_worst / (dist_best + dist_worst)

df["TOPSIS Score"] = scores
df["Rank"] = df["TOPSIS Score"].rank(ascending=False)

print("\nFinal Ranking:")
print(df.sort_values("Rank"))

# Save results
df.to_csv("topsis_results.csv", index=False)

print("\nResults saved to topsis_results.csv")

import matplotlib.pyplot as plt

df_sorted = df.sort_values("Rank")

plt.figure()
plt.bar(df_sorted["Model"], df_sorted["TOPSIS Score"])
plt.xticks(rotation=45)
plt.title("TOPSIS Ranking of Sentence Similarity Models")
plt.xlabel("Models")
plt.ylabel("TOPSIS Score")
plt.tight_layout()
plt.savefig("ranking_plot.png")
plt.show()

