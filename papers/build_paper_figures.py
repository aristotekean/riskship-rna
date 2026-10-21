"""Regenerate paper figures from aggregated full-training artifacts only.

Run from any directory with the project's pinned GP environment.
No data records, category maps or model pickles are read or modified.
"""

import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "results" / "gp_full_v1"
FIGURES = PAPER / "figuras"
NAVY = "#032F49"
TEAL = "#0C5979"
GOLD = "#B98100"


def evolution():
    source = RESULTS / "temporal_180x25_regularized__domain__seed44_history.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    generations = [int(row["generation"]) for row in rows]
    best = [float(row["best_fitness"]) for row in rows]
    mean = [float(row["mean_fitness"]) for row in rows]
    assert generations == list(range(26))
    fig, ax = plt.subplots(figsize=(7.1, 2.65), constrained_layout=True)
    ax.plot(generations, best, color=TEAL, linewidth=2.2, label="Mejor aptitud histórica")
    ax.plot(generations, mean, color=GOLD, linewidth=1.8, label="Media de la población")
    ax.set(xlabel="Generación", ylabel="AP train menos penalización", xlim=(0, 25), ylim=(0.35, 0.75))
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.grid(axis="y", color="#D9E3E9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.savefig(FIGURES / "gp_evolucion_semilla44.png", dpi=300)
    plt.close(fig)


def learned_tree():
    source = RESULTS / "temporal_180x25_regularized__no_domain__seed45.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    root = ast.parse(data["tree"], mode="eval").body
    nodes, edges = [], []
    leaf_count = 0

    def layout(node, depth=0):
        nonlocal leaf_count
        if isinstance(node, ast.Name):
            assert node.id.startswith("x") and node.id[1:].isdigit()
            x = leaf_count
            leaf_count += 1
            name = "$x_{" + node.id[1:] + "}$"
            terminal = True
        elif isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id in {"add", "sub"}
            assert len(node.args) == 2
            children = [layout(child, depth + 1) for child in node.args]
            x = sum(child[0] for child in children) / len(children)
            name = "$+$" if node.func.id == "add" else "$-$"
            terminal = False
            edges.extend(((x, depth), child) for child in children)
        else:
            raise ValueError("Unexpected expression in selected reference tree")
        nodes.append((x, depth, name, terminal))
        return x, depth

    layout(root)
    assert data["seed"] == 45 and len(nodes) == data["nodes"] == 9
    fig, ax = plt.subplots(figsize=(7.1, 2.8), constrained_layout=True)
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [-y1, -y2], color="#7E8F99", linewidth=1.2, zorder=1)
    for x, depth, label, terminal in nodes:
        ax.text(x, -depth, label, ha="center", va="center", fontsize=12,
                color=NAVY, bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EDF4F7" if terminal else "#FFF3CD", "edgecolor": TEAL}, zorder=2)
    ax.set(xlim=(-0.25, leaf_count - 0.75), ylim=(-4.35, 0.35))
    ax.axis("off")
    fig.savefig(FIGURES / "gp_arbol_sin_dominio.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    evolution()
    learned_tree()
    print("Regenerated two figures from full-training aggregate artifacts.")
