"""Build the figures of papers/riskship_anfis.tex.

Run from any directory with the project environment (``uv run python
papers/build_anfis_figures.py``). The comparison chart is drawn from the
aggregated metrics published in README.md ("Protocolo experimental y
resultados (RNA)", "Programación genética (GP)" and "Sistema neuro-difuso
(ANFIS)"); the other figures are the ones the executed notebook
notebooks/anfis_shipment_loss.ipynb exports to informe/figuras/anfis/, copied
here (the membership-function panel is downscaled to keep the repository
small). No model is trained and no data record is read.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parent
NOTEBOOK_FIGURES = PAPER.parent / "informe" / "figuras" / "anfis"
FIGURES = PAPER / "figuras"
NAVY = "#032F49"
TEAL = "#0C5979"
GOLD = "#B98100"
GREY = "#7E8F99"

# Test-set mean and sample standard deviation over five seeds (42-46), same
# temporal holdout for the three models. Sources: README.md tables.
METRICS = ["PR-AUC", "ROC-AUC", "Recall@20 %"]
MODELS = {
    "ANFIS": {"mean": [0.7529, 0.8231, 0.5030], "std": [0.0012, 0.0008, 0.0023], "color": TEAL},
    "RNA": {"mean": [0.7496, 0.8241, 0.5060], "std": [0.0020, 0.0020, 0.0024], "color": NAVY},
    "GP": {"mean": [0.6393, 0.7734, 0.4450], "std": [0.0338, 0.0186, 0.0199], "color": GOLD},
}


def comparison():
    x = np.arange(len(METRICS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.1, 2.9), constrained_layout=True)
    for offset, (name, spec) in zip((-width, 0, width), MODELS.items()):
        bars = ax.bar(x + offset, spec["mean"], width, yerr=spec["std"], capsize=3,
                      color=spec["color"], label=name, error_kw={"ecolor": "#333333", "linewidth": 1})
        for bar, mean in zip(bars, spec["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.012, f"{mean:.3f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Valor en el mes de prueba")
    ax.grid(axis="y", color="#D9E3E9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    fig.savefig(FIGURES / "anfis_rna_gp_comparacion.png", dpi=300)
    plt.close(fig)


def copy_notebook_figures():
    for name, scale in [
        ("anfis_funciones_pertenencia.png", 0.5),
        ("anfis_importancia_permutacion.png", 1.0),
        ("anfis_curva_ganancia.png", 1.0),
    ]:
        source = NOTEBOOK_FIGURES / name
        if not source.exists():
            raise FileNotFoundError(f"{source} missing: run the ANFIS notebook first")
        image = Image.open(source)
        if scale != 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        image.save(FIGURES / name, optimize=True)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    comparison()
    copy_notebook_figures()
    print("Built the comparison chart and copied three notebook figures.")
