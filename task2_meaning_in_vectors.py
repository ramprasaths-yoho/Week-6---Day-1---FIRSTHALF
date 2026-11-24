"""
Week 6 - Day 1 | Task 2: Meaning in Vectors

Setup:
    pip install sentence-transformers pandas matplotlib scikit-learn
"""
import itertools
import re

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the leave policy?",
    "How many casual leaves do I get?",
    "The weather in Chennai is hot",
    "Office timings are 9 to 6",
]
labels = ["S1 password", "S2 login", "S3 leave policy", "S4 casual leaves", "S5 weather", "S6 timings"]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, convert_to_tensor=True)

# ------------------------------------------------------------ 1. 6x6 similarity matrix
sim = util.cos_sim(embeddings, embeddings).cpu().numpy()
sim_df = pd.DataFrame(sim, index=labels, columns=labels).round(3)

print("=" * 70)
print("6x6 COSINE SIMILARITY MATRIX")
print("=" * 70)
for i, s in enumerate(sentences):
    print(f"{labels[i]}: {s}")
print()
print(sim_df.to_string())
sim_df.to_csv("similarity_matrix.csv")

# ------------------------------------------------------------ 2. Most / least similar pairs
pairs = [(i, j, float(sim[i][j])) for i, j in itertools.combinations(range(len(sentences)), 2)]
pairs_sorted = sorted(pairs, key=lambda p: p[2], reverse=True)

print("\n" + "=" * 70)
print("TOP 3 MOST SIMILAR PAIRS")
print("=" * 70)
for rank, (i, j, score) in enumerate(pairs_sorted[:3], start=1):
    print(f"{rank}. {score:.3f} | \"{sentences[i]}\"  <->  \"{sentences[j]}\"")

least = pairs_sorted[-1]
print("\n" + "=" * 70)
print("LEAST SIMILAR PAIR")
print("=" * 70)
print(f"{least[2]:.3f} | \"{sentences[least[0]]}\"  <->  \"{sentences[least[1]]}\"")

# ------------------------------------------------------------ 3. No common words, high similarity
STOPWORDS = {"i", "my", "do", "is", "the", "a", "an", "are", "to", "in", "what", "how", "get", "many"}


def content_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


no_overlap = [
    (i, j, score) for i, j, score in pairs_sorted
    if not (content_words(sentences[i]) & content_words(sentences[j]))
]

print("\n" + "=" * 70)
print("PAIR WITH NO COMMON WORDS BUT HIGH SIMILARITY")
print("=" * 70)
if no_overlap:
    i, j, score = no_overlap[0]
    print(f"\"{sentences[i]}\"  <->  \"{sentences[j]}\"")
    print(f"Shared content words: {content_words(sentences[i]) & content_words(sentences[j]) or 'none'}")
    print(f"Cosine similarity   : {score:.3f}")
    print("Why: the model encodes meaning, not spelling. Both sentences are about being\n"
          "unable to access an account, so their vectors point in nearly the same direction.")

# ------------------------------------------------------------ 4. 2D plot (PCA)
coords = PCA(n_components=2, random_state=42).fit_transform(embeddings.cpu().numpy())

plt.figure(figsize=(9, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=120, c=range(len(sentences)), cmap="tab10")
for (x, y), label, text in zip(coords, labels, sentences):
    plt.annotate(f"{label}\n{text}", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=8)
plt.title("Sentence embeddings (all-MiniLM-L6-v2) reduced to 2D with PCA")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(alpha=0.3)
plt.margins(0.3)
plt.tight_layout()
plt.savefig("embeddings_2d_plot.png", dpi=150)
print("\nSaved: similarity_matrix.csv, embeddings_2d_plot.png")
plt.show()
