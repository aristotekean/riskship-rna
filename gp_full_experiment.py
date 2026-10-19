"""Full-train GP suite. Local data only; selection is sealed before test scoring.

Run: .venv/Scripts/python gp_full_experiment.py --workers 2
Scikit-learn AP is the fitness, cached by expression. No RNA artifact is loaded.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import average_precision_score, precision_recall_curve

from gp_shipment_loss import (CATEGORICAL_FEATURES, DATE_COLUMN, FEATURE_COLUMNS,
                              GROUP, LABEL, NUMERIC_FEATURES, evaluate, preprocess,
                              split_temporal)

SEEDS = [42, 43, 44, 45, 46]


@dataclass(frozen=True)
class Config:
    name: str
    encoding: str
    population: int
    generations: int
    penalty: float = 1e-4
    smoothing: float = 50.0
    max_height: int = 8
    max_nodes: int = 127


# Fixed before looking at any new test predictions. Every fitness uses all train.
CONFIGS = [Config("frequency_100x15", "frequency", 100, 15),
           Config("temporal_130x18", "temporal", 130, 18),
           Config("temporal_180x25_regularized", "temporal", 180, 25, 3e-4)]


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Encoder:
    """Frequencies + optional past-month smoothed label means, numeric features.

    Entire shipments belong to the same month. The target encoding of a row
    sees labels from earlier TRAIN months only. First month uses fixed 0.5.
    Validation/test mappings use all and only TRAIN. Current/future months,
    validation labels, test labels and absolute dates never become predictors.
    Historical labels are assumed observable by the next month (limitation).
    """

    def __init__(self, encoding="temporal", include_domain=True, smoothing=50.0):
        self.encoding = encoding
        self.smoothing = smoothing
        self.categoricals = [c for c in CATEGORICAL_FEATURES
                             if include_domain or c != "DOM_DOMAIN_ID"]
        self.names = ([f"freq__{c}" for c in self.categoricals]
                      + ([f"past_target__{c}" for c in self.categoricals]
                         if encoding == "temporal" else [])
                      + NUMERIC_FEATURES + ["day_of_week"])

    @staticmethod
    def categories(frame, col):
        return frame[col].astype("object").where(frame[col].notna(), "__MISSING__").astype(str)

    def mapping(self, frame, col):
        prior = float(frame[LABEL].mean()) if len(frame) else 0.5
        if not len(frame):
            return {}, prior
        stats = pd.DataFrame({"category": self.categories(frame, col),
                              "label": frame[LABEL]}).groupby("category")["label"].agg(["sum", "count"])
        return ((stats["sum"] + self.smoothing * prior) /
                (stats["count"] + self.smoothing)).to_dict(), prior

    def fit_transform(self, train):
        assert not train[GROUP].isna().any()
        months = train[DATE_COLUMN].dt.to_period("M")
        if (pd.DataFrame({"group": train[GROUP], "month": months})
                .groupby("group")["month"].nunique().max() > 1):
            raise ValueError("Shipment crosses encoding months")
        self.frequencies, self.targets = {}, {}
        for col in self.categoricals:
            self.frequencies[col] = self.categories(train, col).value_counts(normalize=True).to_dict()
            self.targets[col], self.prior = self.mapping(train, col)
        matrix = self.unscaled(train)
        if self.encoding == "temporal":
            offset = len(self.categoricals)
            for month in sorted(months.unique()):
                earlier = train.loc[months < month]
                current = months == month
                for j, col in enumerate(self.categoricals):
                    mapping, prior = self.mapping(earlier, col)
                    matrix[current, offset + j] = self.categories(train.loc[current], col).map(mapping).fillna(prior)
        self.impute = np.nanmedian(matrix, axis=0)
        self.impute = np.nan_to_num(self.impute)
        matrix = np.where(np.isfinite(matrix), matrix, self.impute)
        self.center = np.median(matrix, axis=0)
        self.spread = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
        self.spread[self.spread < 1e-9] = 1.0
        return self.scale(matrix)

    def unscaled(self, frame):
        columns = [self.categories(frame, c).map(self.frequencies[c]).fillna(0.0).to_numpy(float)
                   for c in self.categoricals]
        if self.encoding == "temporal":
            columns += [self.categories(frame, c).map(self.targets[c]).fillna(self.prior).to_numpy(float)
                        for c in self.categoricals]
        columns += [pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
                    for c in NUMERIC_FEATURES]
        columns += [frame["day_of_week"].to_numpy(float)]
        return np.column_stack(columns)

    def scale(self, matrix):
        matrix = np.where(np.isfinite(matrix), matrix, self.impute)
        return np.ascontiguousarray(np.clip((matrix - self.center) / self.spread, -30, 30).T)

    def transform(self, frame):
        return self.scale(self.unscaled(frame))


def bounded(value):
    return np.nan_to_num(np.clip(value, -1e6, 1e6), nan=0.0, posinf=1e6, neginf=-1e6)


def add(a, b): return bounded(a + b)
def sub(a, b): return bounded(a - b)
def mul(a, b): return bounded(a * b)
def pdiv(a, b): return bounded(np.divide(a, np.where(np.abs(b) < 1e-6, 1.0, b)))
def plog(a): return np.sign(a) * np.log1p(np.abs(a))


def primitives(n_features):
    pset = gp.PrimitiveSet("RISK", n_features)
    pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_features)})
    for func, arity, name in [(add, 2, "add"), (sub, 2, "sub"), (mul, 2, "mul"),
                               (pdiv, 2, "pdiv"), (np.minimum, 2, "min"),
                               (np.maximum, 2, "max"), (np.negative, 1, "neg"),
                               (np.abs, 1, "abs"), (np.tanh, 1, "tanh"), (plog, 1, "log")]:
        pset.addPrimitive(func, arity, name=name)
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1, 1))
    return pset


def predict_tree(tree, X):
    pset = primitives(len(X))
    individual = gp.PrimitiveTree.from_string(str(tree), pset)
    func = gp.compile(individual, pset)
    return np.broadcast_to(bounded(func(*X)), X.shape[1:]).copy()


def load_local_model(path):
    """Load a trusted local bundle, including bundles created by Windows spawn.

    Multiprocessing names the main module __mp_main__. Resolving that Encoder
    explicitly makes these original experiment bundles loadable from notebooks
    and other Python programs as well as from the training script.
    """
    class LocalUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in {"__main__", "__mp_main__"} and name == "Encoder":
                return Encoder
            return super().find_class(module, name)

    with open(path, "rb") as stream:
        return LocalUnpickler(stream).load()


def score_raw_shipments(model, raw):
    """Score new item rows; training-cohort filters and labels are not required."""
    frame = raw.copy()
    frame["TIPO_BULKY"] = frame["TIPO_BULKY"].fillna("NONE")
    frame["day_of_week"] = pd.to_datetime(frame[DATE_COLUMN]).dt.dayofweek
    return predict_tree(model["tree"], model["encoder"].transform(frame))


def exact_f1_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Deterministic tie: first maximum, i.e. smallest threshold.
    idx = int(np.argmax(f1))
    return float(thresholds[idx])


def evolution(X, y, X_val, y_val, config, seed, log_path):
    random.seed(seed)
    pset = primitives(len(X))
    if not hasattr(creator, "RiskFullFitness"):
        creator.create("RiskFullFitness", base.Fitness, weights=(1.0,))
        creator.create("RiskFullIndividual", gp.PrimitiveTree, fitness=creator.RiskFullFitness)
    box = base.Toolbox()
    box.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    box.register("individual", tools.initIterate, creator.RiskFullIndividual, box.expr)
    box.register("select", tools.selTournament, tournsize=4)
    box.register("mate", gp.cxOnePoint)
    box.register("expr_mut", gp.genFull, min_=0, max_=2)
    box.register("mutate", gp.mutUniform, expr=box.expr_mut, pset=pset)
    for key, maximum in [(lambda t: t.height, config.max_height), (len, config.max_nodes)]:
        box.decorate("mate", gp.staticLimit(key=key, max_value=maximum))
        box.decorate("mutate", gp.staticLimit(key=key, max_value=maximum))
    cache, candidates = {}, {}

    def fitness(tree):
        expression = str(tree)
        if expression not in cache:
            func = gp.compile(tree, pset)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scores = np.broadcast_to(bounded(func(*X)), y.shape)
            cache[expression] = (float(average_precision_score(y, scores))
                                 - config.penalty * len(tree),)
        return cache[expression]

    box.register("evaluate", fitness)
    population = [box.individual() for _ in range(config.population)]
    # Include individual terminals so evolution can improve simple baselines.
    for i in range(min(len(X), len(population))):
        population[i] = creator.RiskFullIndividual(gp.PrimitiveTree.from_string(f"x{i}", pset))
    hall = tools.HallOfFame(5)
    history = []
    started = time.perf_counter()
    for generation in range(config.generations + 1):
        for tree in population:
            if not tree.fitness.valid:
                tree.fitness.values = box.evaluate(tree)
        hall.update(population)
        candidates[str(hall[0])] = copy.deepcopy(hall[0])
        values = [tree.fitness.values[0] for tree in population]
        row = {"generation": generation, "best_fitness": hall[0].fitness.values[0],
               "population_best_fitness": max(values), "mean_fitness": float(np.mean(values)),
               "best_nodes": len(hall[0]), "best_height": hall[0].height,
               "unique_fitness_evaluations": len(cache), "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        # Aggregate-only live progress survives interrupted runs.
        pd.DataFrame(history).to_csv(log_path, index=False)
        if generation == config.generations:
            break
        elite = copy.deepcopy(hall[0])
        population = algorithms.varAnd(box.select(population, config.population - 1), box, .8, .2)
        population.append(elite)
    for tree in hall:
        candidates[str(tree)] = tree
    scored = [(float(average_precision_score(y_val, predict_tree(expr, X_val))),
               -len(tree), expr, tree) for expr, tree in candidates.items()]
    selected_ap, _, selected, tree = max(scored, key=lambda r: (r[0], r[1], r[2]))
    return {"tree": selected, "nodes": len(tree), "height": tree.height,
            "training_ap": tree.fitness.values[0] + config.penalty * len(tree),
            "validation_ap": selected_ap, "candidate_count": len(scored),
            "unique_fitness_evaluations": len(cache), "fit_seconds": time.perf_counter() - started}


def load_parts(data):
    raw = pd.read_parquet(data)
    clean = preprocess(raw)
    train, val, test, summary = split_temporal(clean)
    expected = [(125082, 112303), (18831, 17297), (21089, 19506)]
    observed = [(len(p), p[GROUP].nunique()) for p in (train, val, test)]
    if observed != expected or len(clean) != 165002:
        raise ValueError(f"Split differs from frozen RNA reference: {observed}")
    if raw.shape != (262317, 58):
        raise ValueError(f"Dataset differs from original: {raw.shape}")
    if raw.groupby(GROUP)[LABEL].nunique().max() != 1:
        raise ValueError("Shipment label conflicts")
    return train, val, test, summary


def fit_job(job):
    data, output_dir, config_dict, seed, include_domain = job
    config = Config(**config_dict)
    name = f"{config.name}__{'domain' if include_domain else 'no_domain'}__seed{seed}"
    directory = Path(output_dir)
    summary_path = directory / f"{name}.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    train, val, _, split = load_parts(data)
    encoder = Encoder(config.encoding, include_domain, config.smoothing)
    X = encoder.fit_transform(train)
    X_val = encoder.transform(val)
    y, y_val = train[LABEL].to_numpy(int), val[LABEL].to_numpy(int)
    result = evolution(X, y, X_val, y_val, config, seed, directory / f"{name}_history.csv")
    val_scores = predict_tree(result["tree"], X_val)
    threshold = exact_f1_threshold(y_val, val_scores)
    # Local model includes category mappings: confidential, excluded from share ZIP.
    bundle = {"encoder": encoder, "tree": result["tree"], "threshold": threshold,
              "score_is_probability": False, "config": asdict(config), "seed": seed}
    private = directory / "private_models"
    private.mkdir(exist_ok=True)
    model_path = private / f"{name}.pickle"
    with model_path.open("wb") as stream:
        pickle.dump(bundle, stream)
    with model_path.open("rb") as stream:
        restored = pickle.load(stream)
    if not np.array_equal(val_scores, predict_tree(restored["tree"], restored["encoder"].transform(val))):
        raise AssertionError("Model round-trip mismatch")
    result.update({"run": name, "config": asdict(config), "seed": seed,
                   "include_domain": include_domain, "fitness_rows": len(train),
                   "terminal_names": dict(enumerate(encoder.names)),
                   "split": split, "threshold": threshold,
                   "validation": evaluate(y_val, val_scores, threshold),
                   "total_seconds": time.perf_counter() - started,
                   "model_path": str(model_path.relative_to(directory))})
    json_write(summary_path, result)
    return result


def fit_batch(jobs, workers):
    if workers == 1:
        # Also supports the self-contained notebook's embedded modules.
        results = [fit_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(fit_job, jobs))
    for result in results:
        print(f"{result['run']}: train AP={result['training_ap']:.4f}, "
              f"validation AP={result['validation_ap']:.4f}, "
              f"{result['fit_seconds']:.1f}s, {result['nodes']} nodes", flush=True)
    return results


def full_suite(data, directory, workers=2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = {"configurations": [asdict(c) for c in CONFIGS], "screen_seeds": SEEDS[:2],
            "final_seeds": SEEDS, "fitness": "all 125082 TRAIN rows, AP - lambda * nodes",
            "selection": "highest mean validation AP over screen seeds; then five repetitions",
            "tree_selection": "highest validation AP among per-generation train champions + final train HallOfFame(5)",
            "test_policy": "all choices and thresholds sealed before any new test scoring",
            "dataset_sha256": sha256(data), "score_is_probability": False,
            "previous_test_exposure": "Original small GP and RNA test results were already visible; this is a fixed comparative holdout, not a fresh blind test.",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "logical_cpus": os.cpu_count(), "workers": workers,
                            "packages": {p: importlib.metadata.version(p) for p in
                                         ["numpy", "pandas", "scikit-learn", "deap", "pyarrow", "matplotlib"]}}}
    plan_path = directory / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError("Existing plan differs; use a new experiment directory")
    else:
        json_write(plan_path, plan)
    jobs = [(str(data), str(directory), asdict(c), seed, True) for c in CONFIGS for seed in SEEDS[:2]]
    print("Screening three fixed configurations, full TRAIN fitness, validation only.", flush=True)
    screen = fit_batch(jobs, workers)
    means = {c.name: float(np.mean([r["validation_ap"] for r in screen if r["config"]["name"] == c.name]))
             for c in CONFIGS}
    best = max(CONFIGS, key=lambda c: means[c.name])
    json_write(directory / "configuration_selection.json", {"validation_means": means, "selected": asdict(best)})
    print(f"Selected using validation: {best.name}", flush=True)
    extra = fit_batch([(str(data), str(directory), asdict(best), s, True) for s in SEEDS[2:]], workers)
    final = [r for r in screen if r["config"]["name"] == best.name] + extra
    ablation = fit_batch([(str(data), str(directory), asdict(best), s, False) for s in SEEDS], workers)
    reference = max(final, key=lambda r: r["validation_ap"])
    reference_no_domain = max(ablation, key=lambda r: r["validation_ap"])
    # At this point no final run has computed test scores or metrics.
    sealed = {"configuration": asdict(best), "reference_run": reference["run"],
              "reference_no_domain": reference_no_domain["run"],
              "evaluation_runs": [r["run"] for r in final + ablation],
              "thresholds": {r["run"]: r["threshold"] for r in final + ablation},
              "test_used_for_selection": False}
    json_write(directory / "selection_sealed_before_test.json", sealed)
    print("Selection sealed. Starting final TEST evaluation.", flush=True)
    train, val, test, split = load_parts(data)
    records = []
    curves = {}
    for result in final + ablation:
        with (directory / result["model_path"]).open("rb") as stream:
            bundle = pickle.load(stream)
        scores = predict_tree(bundle["tree"], bundle["encoder"].transform(test))
        metrics = evaluate(test[LABEL].to_numpy(int), scores, bundle["threshold"])
        result["test"] = metrics
        json_write(directory / f"{result['run']}.json", result)
        records.append({"run": result["run"], "seed": result["seed"],
                        "include_domain": result["include_domain"], "config": best.name,
                        "fit_seconds": result["fit_seconds"], "nodes": result["nodes"],
                        "height": result["height"], "train_pr_auc": result["training_ap"],
                        "validation_pr_auc": result["validation_ap"], "threshold": result["threshold"],
                        **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        if result["run"] in [reference["run"], reference_no_domain["run"]]:
            curves[result["include_domain"]] = (scores, result)
    table = pd.DataFrame(records)
    table.to_csv(directory / "final_metrics.csv", index=False)
    summary = {}
    numeric = [c for c in table.select_dtypes(include="number").columns if c != "seed"]
    for included, group in table.groupby("include_domain"):
        summary["with_domain" if included else "without_domain"] = {
            c: {"mean": float(group[c].mean()), "std": float(group[c].std(ddof=1))} for c in numeric}
    json_write(directory / "summary.json", summary)
    make_figures(directory, curves, test[LABEL].to_numpy(int), final + ablation)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def make_figures(directory, curves, y, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for included, (scores, result) in curves.items():
        name = "Con DOM_DOMAIN_ID" if included else "Sin DOM_DOMAIN_ID"
        p, r, _ = precision_recall_curve(y, scores)
        axes[0].plot(r, p, label=f"{name}: AP={result['test']['pr_auc']:.3f}")
        k = [1, 5, 10, 20]
        axes[1].plot(k, [result["test"][f"recall_at_{x}%"] for x in k], marker="o", label=name)
    axes[0].axhline(y.mean(), color="gray", ls="--", label="Azar")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Curva PR: semilla elegida en validación")
    axes[1].plot([1, 5, 10, 20], [.01, .05, .1, .2], color="gray", ls="--", label="Azar")
    axes[1].set(xlabel="Artículos revisados (%)", ylabel="Pérdidas capturadas", title="Recall@K en test")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "pr_and_ranking.png", dpi=180)
    plt.close(fig)
    for included, (_, result) in curves.items():
        label = "with_domain" if included else "without_domain"
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(np.array(result["test"]["confusion_matrix"]),
                               display_labels=["No perdido", "Perdido"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"GP {label}, semilla {result['seed']}")
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{label}.png", dpi=180)
        plt.close(fig)
        tree_figure(result, figures / f"tree_{label}.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for result in results:
        history = pd.read_csv(directory / f"{result['run']}_history.csv")
        name = f"{'Con' if result['include_domain'] else 'Sin'} DOM, {result['seed']}"
        axes[0].plot(history.generation, history.best_fitness, label=name)
        axes[1].plot(history.generation, history.best_nodes, label=name)
    axes[0].set(xlabel="Generación", ylabel="AP train - penalización", title="Evolución del fitness (train completo)")
    axes[1].set(xlabel="Generación", ylabel="Nodos", title="Complejidad del campeón de train")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "evolution.png", dpi=180)
    plt.close(fig)


def tree_figure(result, destination):
    import matplotlib.pyplot as plt
    tree = gp.PrimitiveTree.from_string(result["tree"], primitives(len(result["terminal_names"])))
    names = {f"x{i}": n for i, n in result["terminal_names"].items()}
    nodes, edges, labels = gp.graph(tree)
    children = {i: [] for i in nodes}
    for parent, child in edges:
        children[parent].append(child)
    positions, leaf = {}, [0]

    def place(node, depth=0):
        if not children[node]:
            x = leaf[0]
            leaf[0] += 1
        else:
            x = float(np.mean([place(c, depth + 1) for c in children[node]]))
        positions[node] = (x, -depth)
        return x

    place(0)
    fig, ax = plt.subplots(figsize=(min(26, max(10, leaf[0] * 2.3)), max(4, tree.height * 1.6)))
    for a, b in edges:
        ax.plot([positions[a][0], positions[b][0]], [positions[a][1], positions[b][1]], color="#777", zorder=1)
    for node in nodes:
        text = names.get(str(labels[node]), str(labels[node]))
        text = text.replace("past_target__", "TE pasado\n").replace("freq__", "Frecuencia\n")
        ax.text(*positions[node], text, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=.4", fc="#e5efff", ec="#3483fa"), zorder=2)
    ax.margins(.12)
    ax.axis("off")
    ax.set_title(f"Árbol GP elegido en validación: {result['nodes']} nodos, altura {result['height']}")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_full_v1"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    full_suite(args.data, args.output, args.workers)


if __name__ == "__main__":
    main()
