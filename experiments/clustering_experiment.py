import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

MODEL = "all-mpnet-base-v2"
THRESHOLD = 0.75

SENTENCES = [
    "The cat sat on the mat",
    "Dogs love to play fetch",
    "My feline friend enjoys napping",
    "The puppy ran across the yard",
    "Machine learning is fascinating",
    "AI models can process text",
]

os.makedirs("experiments/dendrograms", exist_ok=True)

# Encode sentences
model = SentenceTransformer(MODEL)
embeddings = model.encode(SENTENCES)

# Create dendrogram
Z = linkage(pdist(embeddings, metric='cosine'), method='average')

plt.figure(figsize=(12, 6))
labels = [s[:35] + "..." if len(s) > 35 else s for s in SENTENCES]
dendrogram(Z, labels=labels, leaf_rotation=45, color_threshold=THRESHOLD)
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold: {THRESHOLD}')
plt.legend()
plt.title(f"{MODEL} | threshold={THRESHOLD}")
plt.ylabel("Cosine Distance")
plt.tight_layout()

filename = f"experiments/dendrograms/{MODEL.replace('/', '_')}_t{THRESHOLD}.png"
plt.savefig(filename, dpi=150)
plt.close()
print(f"Dendrogram saved: {filename}")

# Get groups
clustering = AgglomerativeClustering(
    n_clusters=None, distance_threshold=THRESHOLD, metric='cosine', linkage='average'
)
labels_pred = clustering.fit_predict(embeddings)

groups = {}
for sentence, label in zip(SENTENCES, labels_pred):
    groups.setdefault(label, []).append(sentence)

print(f"\nGroups ({len(groups)} clusters):")
for i, group in enumerate(groups.values(), 1):
    print(f"\nGroup {i}:")
    for s in group:
        print(f"  - {s}")
