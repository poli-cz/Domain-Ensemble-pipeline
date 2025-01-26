import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define positions for blocks and arrows
positions = {
    "Lexical Features": (0, 2),
    "TLS Features": (0, 1),
    "Visual Features": (0, 0),
    "RNN (Lexical Analysis)": (2, 2),
    "CNN (Visual Data)": (2, 0),
    "TLS Classifier (Attention)": (2, 1),
    "Meta-Classifier (XGBoost)": (4, 1),
    "Final Prediction": (6, 1),
}

# Draw rectangles for components
for label, (x, y) in positions.items():
    ax.add_patch(mpatches.FancyBboxPatch((x, y - 0.3), 1.8, 0.6, boxstyle="round,pad=0.1", color="lightblue"))
    ax.text(x + 0.9, y, label, ha="center", va="center", fontsize=9, weight="bold")

# Draw arrows between components
arrows = [
    ("Lexical Features", "RNN (Lexical Analysis)"),
    ("TLS Features", "TLS Classifier (Attention)"),
    ("Visual Features", "CNN (Visual Data)"),
    ("RNN (Lexical Analysis)", "Meta-Classifier (XGBoost)"),
    ("TLS Classifier (Attention)", "Meta-Classifier (XGBoost)"),
    ("CNN (Visual Data)", "Meta-Classifier (XGBoost)"),
    ("Meta-Classifier (XGBoost)", "Final Prediction"),
]

for start, end in arrows:
    start_x, start_y = positions[start]
    end_x, end_y = positions[end]
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x + 1.8, start_y),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    )

# Set limits and remove axes
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 3)
ax.axis("off")

# Show the plot
plt.title("Architecture of the Composite Classifier", fontsize=14, weight="bold")
plt.show()
