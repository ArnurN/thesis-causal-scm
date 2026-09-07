import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" 🧠 PHASE 3A: CAUSAL GRAPH (DAG) DECLARATION")
print("="*80)
print("Formally defining assumptions before Double Machine Learning...\n")

# 1. Initialize the Directed Acyclic Graph
G = nx.DiGraph()

# 2. Define the Nodes (Variables)
nodes = {
    'T': 'Price\n(Treatment)',
    'Y': 'Demand\n(Outcome)',
    'W1': 'Holidays &\nSNAP (W)',
    'W2': 'Seasonality\n(wday, month) (W)',
    'W3': 'Historical\nVolatility (W)',
    'X': 'Product &\nStore ID (X)'
}

# Add nodes to graph
for node, label in nodes.items():
    G.add_node(node, label=label)

# 3. Define the Causal Arrows (Edges)
edges = [
    ('T', 'Y'),   # The pure Elasticity we want to measure
    ('W1', 'T'), ('W1', 'Y'),  # Holidays cause both Price & Demand
    ('W2', 'T'), ('W2', 'Y'),  # Seasonality causes both
    ('W3', 'T'), ('W3', 'Y'),  # Past Volatility causes both
    ('X', 'T'), ('X', 'Y')     # Product type affects baseline price and baseline demand
]

G.add_edges_from(edges)

# 4. Visualization Architecture
plt.figure(figsize=(10, 6))

# Strictly define where the nodes sit on the screen for academic clarity
pos = {
    'W1': (0.2, 0.9),
    'W2': (0.5, 0.9),
    'W3': (0.8, 0.9),
    'T': (0.2, 0.3),
    'Y': (0.8, 0.3),
    'X': (0.5, 0.1)
}

# Draw Nodes
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')

# Draw Arrows (Edges)
# Highlight the main causal arrow (Price -> Demand) in Green
edge_colors = ['green' if edge == ('T', 'Y') else 'gray' for edge in G.edges()]
edge_widths = [3.0 if edge == ('T', 'Y') else 1.5 for edge in G.edges()]

nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', 
                       arrowsize=20, edge_color=edge_colors, width=edge_widths)

# Draw Labels
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

plt.title("Formal Causal DAG: Isolating Price Elasticity (Double Machine Learning)", fontsize=14, fontweight='bold')
plt.axis('off')

# Save and Show the Graph
plt.tight_layout()
plt.savefig("causal_dag_model.png", dpi=300)
print("✅ Causal DAG successfully saved as 'causal_dag_model.png'.")
print("✅ The pure causal path (Price -> Demand) is highlighted in GREEN.")
print("✅ Confounding backdoor paths (W -> T, W -> Y) are formally declared.")
plt.show()

print("\n" + "="*80)