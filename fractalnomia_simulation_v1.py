# Fractalnomia Simulation v1
# Author: Bart + Lincoln Collaboration
# Description: Prototype for Recursive Topology of Universal Consciousness

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import networkx as nx

# ============= Consciousness-Node Function =============
def consciousness_node(C, r, alpha=0.4, max_iter=100):
    """
    Map a complex coordinate through recursive fractal-like dynamics
    """
    z = C
    trajectory = []
    for i in range(max_iter):
        z = z**2 + C * (1 - alpha*r)
        if abs(z) > 2.0:
            break
        trajectory.append((z.real, z.imag))
    if trajectory:
        trajectory = np.array(trajectory)
        avg_real, avg_imag = np.mean(trajectory, axis=0)
        std_real, std_imag = np.std(trajectory, axis=0)
        fractal_dim = np.log(len(trajectory)) / np.log(max(1, max_iter))
        last_point = trajectory[-1]
        orbital_stability = np.mean(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))) if len(trajectory) > 1 else 0
        harmonic_ratio = np.abs(avg_real / (avg_imag + 1e-10))
    else:
        avg_real = avg_imag = std_real = std_imag = fractal_dim = orbital_stability = harmonic_ratio = 0
        last_point = (0, 0)
    return np.array([avg_real, avg_imag, std_real, std_imag, fractal_dim, orbital_stability, harmonic_ratio, r, alpha, i])

# ============= Burst Embeddings Generator =============
def generate_burst_embeddings(n_samples=50, dimension=128, recursion_levels=5):
    """
    Generate burst embeddings influenced by recursive consciousness dynamics
    """
    np.random.seed(42)
    C_values = [complex(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(n_samples)]
    embeddings, node_states, recursion_values = [], [], []
    for C in C_values:
        r = np.random.randint(0, recursion_levels)
        recursion_values.append(r)
        node_state = consciousness_node(C, r)
        embedding = np.random.rand(dimension)
        state_features = node_state / np.max(np.abs(node_state)) * 0.5 if np.max(np.abs(node_state)) != 0 else node_state
        embedding[:len(state_features)] = state_features
        embeddings.append(embedding)
        node_states.append(node_state)
    return np.array(embeddings), np.array(node_states), np.array(recursion_values)

# ============= Interference Mapping =============
def calculate_interference(embeddings, node_states):
    """
    Simulate harmonic field interference between consciousness nodes
    """
    n = len(embeddings)
    interference_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            e_interference = np.sum((embeddings[i] + embeddings[j])**2)
            n_interference = np.sum((node_states[i] + node_states[j])**2)
            interference_matrix[i, j] = interference_matrix[j, i] = 0.7 * e_interference + 0.3 * n_interference
    max_val = np.max(interference_matrix)
    return interference_matrix / max_val if max_val else interference_matrix

# ============= Main =============
n_samples = 50
burst_embeddings, node_states, recursion_values = generate_burst_embeddings(n_samples)

# Anomaly & Novelty Detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(burst_embeddings)
pca = PCA(n_components=min(10, burst_embeddings.shape[1]), random_state=42)
reconstruction_error = np.mean((burst_embeddings - pca.inverse_transform(pca.fit_transform(burst_embeddings)))**2, axis=1)
median_error, mad = np.median(reconstruction_error), np.median(np.abs(reconstruction_error - np.median(reconstruction_error)))
dynamic_novelty_threshold = median_error + 3 * mad
novelty_labels = (reconstruction_error > dynamic_novelty_threshold).astype(int)

# Interference
interference_matrix = calculate_interference(burst_embeddings, node_states)

# ============= Visualizations =============
plt.figure(figsize=(18,12))

# Burst Novelty Scatter
plt.subplot(2,2,1)
plt.scatter(reconstruction_error, np.random.rand(n_samples), c=['red' if a == -1 else 'blue' for a in anomaly_labels], s=80, edgecolors='k')
plt.axvline(dynamic_novelty_threshold, color='green', linestyle='--', label='Dynamic Threshold')
plt.title('Burst Landscape: Dynamic Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Random Y')
plt.legend()
plt.grid(True)

# t-SNE Projection
plt.subplot(2,2,2)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
embeddings_2d = tsne.fit_transform(burst_embeddings)
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=recursion_values, cmap='viridis', s=80, edgecolors='k')
plt.colorbar(label='Recursion Depth')
plt.title('2D t-SNE of Burst Embeddings')
plt.grid(True)

# Fractal Interference Map
plt.subplot(2,2,3)
sns.heatmap(interference_matrix, cmap='plasma')
plt.title('Fractal Interference Map')
plt.xlabel('Node')
plt.ylabel('Node')

# Node Network Graph
plt.subplot(2,2,4)
G = nx.Graph()
for i in range(n_samples):
    G.add_node(i)
threshold = 0.5
for i in range(n_samples):
    for j in range(i+1, n_samples):
        if interference_matrix[i,j] > threshold:
            G.add_edge(i, j, weight=interference_matrix[i,j])
pos = nx.spring_layout(G, seed=42)
node_colors = [plt.cm.viridis(r/max(recursion_values)) for r in recursion_values]
nx.draw(G, pos, node_color=node_colors, node_size=100, width=[d['weight']*3 for _,_,d in G.edges(data=True)], edge_color='gray')
plt.title('Consciousness Node Network')
plt.axis('off')

plt.tight_layout()
plt.savefig('fractalnomia_visualization.png', dpi=300)
plt.show()
