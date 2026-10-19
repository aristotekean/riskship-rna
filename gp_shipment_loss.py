"""RiskShip symbolic-ranking GP experiment.

This is an additive experiment: it never loads or overwrites the RNA model.
The default command is deliberately a small smoke test.  It uses the same
label, row filters, temporal split, and shipment-group audit as the RNA
notebook, then evolves real DEAP expression trees whose output is a ranking
score (not a calibrated probability).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABEL = "is_lost"
GROUP = "SHIPMENT_ID"
DATE_COLUMN = "date_status_0"
MATURITY_RATIO = 0.5
MIN_HOLDOUT_SHARE = 0.10
DAY_NAMES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
CATEGORICAL_FEATURES = [
    "PICKING_TYPE", "SHP_LG_FACILITY_ID", "DOM_DOMAIN_ID",
    "ORD_CATEGORY_NAME_L1", "BRAND_NAME", "TIPO_BULKY",
]
NUMERIC_FEATURES = [
    "PESO_TOTAL_KG", "DIM_MAX_CM", "VALOR_TOTAL_USD", "CANTIDAD_ITEMS",
    "RATIO_SVC_MES_ANTERIOR", "ES_BULKY",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["day_of_week"]
MISSINGNESS_ARTIFACT_COLUMNS = [
    "PESO_TOTAL_KG", "DIM_MAX_CM", "VALOR_TOTAL_USD", "CANTIDAD_ITEMS",
]


def protected_div(left, right):
    return np.divide(left, np.where(np.abs(right) < 1e-6, 1.0, right))


def protected_log(value):
    return np.sign(value) * np.log1p(np.abs(value))


def protected_tanh(value):
    return np.tanh(np.clip(value, -20.0, 20.0))


def protected_clip(value):
    return np.nan_to_num(np.clip(value, -1e6, 1e6), nan=0.0, posinf=1e6, neginf=-1e6)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    artifact_null = frame[MISSINGNESS_ARTIFACT_COLUMNS + [DATE_COLUMN]].isna().any(axis=1)
    frame = frame.loc[~artifact_null].copy()
    month = frame[DATE_COLUMN].dt.to_period("M")
    month_stats = frame.groupby(month)[LABEL].agg(["nunique", "mean"])
    valid_months = list(month_stats.index[month_stats["nunique"] == 2])
    maturity_floor = MATURITY_RATIO * month_stats.loc[valid_months, "mean"].median()
    while valid_months and month_stats.loc[valid_months[-1], "mean"] < maturity_floor:
        valid_months.pop()
    frame = frame.loc[month.isin(valid_months)].copy()
    frame["TIPO_BULKY"] = frame["TIPO_BULKY"].fillna("NONE")
    frame["day_of_week"] = frame[DATE_COLUMN].dt.dayofweek
    return frame[FEATURE_COLUMNS + [LABEL, GROUP, DATE_COLUMN]]


def split_temporal(clean: pd.DataFrame):
    month = clean[DATE_COLUMN].dt.to_period("M")
    shares = month.value_counts(normalize=True).sort_index()
    available = list(shares.index)

    def take_latest(months):
        taken = []
        while months and shares.loc[taken].sum() < MIN_HOLDOUT_SHARE:
            taken.insert(0, months.pop())
        return taken

    test_months = take_latest(available)
    val_months = take_latest(available)
    train_months = available
    train = clean.loc[month.isin(train_months)].copy()
    val = clean.loc[month.isin(val_months)].copy()
    test = clean.loc[month.isin(test_months)].copy()
    sets = [set(train[GROUP]), set(val[GROUP]), set(test[GROUP])]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    summary = {
        "train": {"months": [str(x) for x in train_months], "rows": len(train), "shipments": train[GROUP].nunique(), "positive_rate": float(train[LABEL].mean())},
        "validation": {"months": [str(x) for x in val_months], "rows": len(val), "shipments": val[GROUP].nunique(), "positive_rate": float(val[LABEL].mean())},
        "test": {"months": [str(x) for x in test_months], "rows": len(test), "shipments": test[GROUP].nunique(), "positive_rate": float(test[LABEL].mean())},
        "shared_shipment_groups": 0,
    }
    return train, val, test, summary


class TrainOnlyFrequencyEncoder:
    """Encode categories by train frequencies; unknown validation/test = 0."""

    def __init__(self):
        self.maps = {}

    def fit(self, frame: pd.DataFrame):
        for column in CATEGORICAL_FEATURES:
            values = frame[column].fillna("__MISSING__").astype(str)
            self.maps[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in CATEGORICAL_FEATURES:
            values = frame[column].fillna("__MISSING__").astype(str)
            columns.append(values.map(self.maps[column]).fillna(0.0).to_numpy(dtype=float))
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            columns.append(values)
        columns.append(frame["day_of_week"].to_numpy(dtype=float) / 6.0)
        result = np.column_stack(columns)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_train_only(train_x, *others):
    center = np.median(train_x, axis=0)
    spread = np.subtract(*np.percentile(train_x, [75, 25], axis=0))
    spread = np.where(spread < 1e-9, 1.0, spread)
    return [(x - center) / spread for x in (train_x, *others)]


def make_toolbox(n_features: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    pset = gp.PrimitiveSet("RISK", n_features)
    pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_features)})
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="pdiv")
    pset.addPrimitive(np.minimum, 2, name="min")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(protected_tanh, 1, name="tanh")
    pset.addPrimitive(protected_log, 1, name="log")
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1.0, 1.0))
    if not hasattr(creator, "FitnessMaxRiskShip"):
        creator.create("FitnessMaxRiskShip", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualRiskShip"):
        creator.create("IndividualRiskShip", gp.PrimitiveTree, fitness=creator.FitnessMaxRiskShip)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.IndividualRiskShip, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=lambda tree: tree.height, max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda tree: tree.height, max_value=8))
    return toolbox, pset


def evolve(X, y, population_size, generations, seed, penalty):
    toolbox, pset = make_toolbox(X.shape[1], seed)

    def fitness(individual):
        try:
            func = toolbox.compile(expr=individual)
            score = protected_clip(func(*X.T))
            if np.unique(score).size < 2 or not np.isfinite(score).all():
                return (-1.0,)
            value = average_precision_score(y, score) - penalty * len(individual)
            return (float(value),)
        except (ArithmeticError, FloatingPointError, ValueError, TypeError, OverflowError):
            return (-1.0,)

    toolbox.register("evaluate", fitness)
    population = toolbox.population(n=population_size)
    history = []
    hall = tools.HallOfFame(1)
    for generation in range(generations + 1):
        invalid = [item for item in population if not item.fitness.valid]
        for item in invalid:
            item.fitness.values = toolbox.evaluate(item)
        hall.update(population)
        values = [item.fitness.values[0] for item in population]
        history.append({
            "generation": generation,
            "best_fitness": float(max(values)),
            "mean_fitness": float(np.mean(values)),
            "best_nodes": len(hall[0]),
            "best_height": hall[0].height,
        })
        if generation == generations:
            break
        offspring = algorithms.varAnd(
            toolbox.select(population, len(population)), toolbox,
            cxpb=0.8, mutpb=0.2,
        )
        population = offspring
    best = hall[0]
    predictor = toolbox.compile(expr=best)
    return best, predictor, history, str(best), pset


def choose_threshold(y, scores):
    candidates = np.unique(scores)
    if len(candidates) > 1000:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 1000))
    best = max(
        ((f1_score(y, scores >= threshold, zero_division=0), float(threshold)) for threshold in candidates),
        key=lambda pair: pair[0],
    )
    return best[1], best[0]


def ranking_metrics(y, scores):
    result = {}
    order = np.argsort(-scores, kind="stable")
    for fraction in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(round(fraction * len(y))))
        hits = int(np.asarray(y)[order[:k]].sum())
        result[f"recall_at_{fraction:.0%}"] = hits / float(np.sum(y))
        result[f"precision_at_{fraction:.0%}"] = hits / k
    return result


def evaluate(y, scores, threshold):
    predicted = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
        **ranking_metrics(y, scores),
    }


def run(args):
    started = time.perf_counter()
    np.random.seed(args.seed)
    random.seed(args.seed)
    raw = pd.read_parquet(args.data)
    clean = preprocess(raw)
    train, val, test, split_summary = split_temporal(clean)
    encoder = TrainOnlyFrequencyEncoder().fit(train)
    X_train, X_val, X_test = standardize_train_only(
        *[encoder.transform(frame) for frame in (train, val, test)]
    )
    y_train, y_val, y_test = [frame[LABEL].to_numpy(dtype=int) for frame in (train, val, test)]
    rng = np.random.default_rng(args.seed)
    if args.fitness_sample < len(X_train):
        indices = rng.choice(len(X_train), size=args.fitness_sample, replace=False)
        X_fit, y_fit = X_train[indices], y_train[indices]
    else:
        X_fit, y_fit = X_train, y_train
    best, predictor, history, tree, pset = evolve(
        X_fit, y_fit, args.population, args.generations, args.seed, args.penalty
    )
    score_fn = lambda X: protected_clip(predictor(*X.T))
    val_scores, test_scores = score_fn(X_val), score_fn(X_test)
    threshold, val_f1 = choose_threshold(y_val, val_scores)
    output = {
        "experiment": "gp_smoke_test" if args.population <= 60 and args.generations <= 5 else "gp_experiment",
        "seed": args.seed,
        "data": {"path_used": str(args.data), "raw_rows": len(raw), "clean_rows": len(clean), "fitness_rows": len(X_fit)},
        "split": split_summary,
        "preprocessing": {
            "categorical_encoding": "train-only frequency encoding; unknown=0",
            "numeric_scaling": "train-only median/IQR",
            "score_is_probability": False,
            "features": FEATURE_COLUMNS,
            "dom_domain_id_included": "DOM_DOMAIN_ID" in FEATURE_COLUMNS,
        },
        "gp": {"population": args.population, "generations": args.generations, "crossover_probability": 0.8, "mutation_probability": 0.2, "tournament_size": 4, "max_height": 8, "penalty": args.penalty, "fitness": "average_precision - penalty * node_count", "best_tree": tree, "best_nodes": len(best), "best_height": best.height, "primitive_set": str(pset)},
        "validation": {"selected_threshold_on_validation": threshold, "f1_at_selected_threshold": val_f1, **evaluate(y_val, val_scores, threshold)},
        "test": evaluate(y_test, test_scores, threshold),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(history).to_csv(args.output.with_name(args.output.stem + "_history.csv"), index=False)
    args.output.with_name(args.output.stem + "_tree.txt").write_text(tree, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "clean_rows": len(clean), "split": split_summary, "tree": tree, "validation": output["validation"], "test": output["test"], "elapsed_seconds": output["elapsed_seconds"]}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_smoke_results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--fitness-sample", type=int, default=2500)
    parser.add_argument("--penalty", type=float, default=1e-4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
