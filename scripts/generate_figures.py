"""Reproduce the evaluation figures used in the README.

Mirrors the pipeline in model.ipynb (regex cleaning -> CountVectorizer ->
MultinomialNB, 80/20 split) with a fixed seed and stratification so the
numbers are repeatable.

    python scripts/generate_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
SEED = 42

INK, MUTED, ACCENT, COPPER, SAGE, LINE = "#202c3a", "#536171", "#305b8a", "#b47b40", "#5f8a6e", "#dce1e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

NAMES = {"Portugeese": "Portuguese", "Sweedish": "Swedish"}
GROUPS = {
    "European": ["English", "French", "Spanish", "Portuguese", "Italian", "Russian", "Swedish", "Dutch", "German", "Danish", "Greek"],
    "South Asian": ["Malayalam", "Hindi", "Tamil", "Kannada", "Urdu"],
    "Middle East & Anatolia": ["Arabic", "Turkish"],
}
GROUP_COLOR = {"European": ACCENT, "South Asian": COPPER, "Middle East & Anatolia": SAGE}
LANG_GROUP = {lang: g for g, langs in GROUPS.items() for lang in langs}


def clean(text: str) -> str:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"\[\]", " ", text)
    return text.lower()


def main() -> None:
    df = pd.read_csv(ROOT / "language_detection.csv")
    df["Language"] = df["Language"].replace(NAMES)
    df["clean"] = df["Text"].astype(str).map(clean)

    x_tr, x_te, y_tr, y_te = train_test_split(
        df["clean"], df["Language"], test_size=0.2, stratify=df["Language"], random_state=SEED
    )
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)

    cv_scores = cross_val_score(
        make_pipeline(CountVectorizer(), MultinomialNB()), df["clean"], df["Language"],
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy",
    )
    vocab = len(model.named_steps["countvectorizer"].vocabulary_)
    metrics = {
        "samples": int(len(df)), "languages": int(df["Language"].nunique()), "vocabulary": vocab,
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_te, pred, average="weighted"), 4),
        "cv5_mean": round(cv_scores.mean(), 4), "cv5_std": round(cv_scores.std(), 4),
    }

    # 1. Dataset composition
    counts = df["Language"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(counts.index, counts.values, color=[GROUP_COLOR[LANG_GROUP[l]] for l in counts.index], width=0.72)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 18, f"{v:,}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_ylabel("Sentences")
    ax.set_title("Corpus composition by language", loc="left")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    ax.legend(handles, GROUP_COLOR.keys(), frameon=False, loc="upper right")
    fig.savefig(OUT / "dataset_distribution.png")
    plt.close(fig)

    # 2. Normalised confusion matrix, ordered by language group
    order = [l for g in GROUPS.values() for l in g]
    cm = confusion_matrix(y_te, pred, labels=order).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("ed", ["#ffffff", "#dbe5f0", ACCENT, INK])
    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm_norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}" if v < 0.995 else "1", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 0.5 else INK)
    for b in (11, 16):
        ax.axhline(b - 0.5, color=COPPER, lw=1)
        ax.axvline(b - 0.5, color=COPPER, lw=1)
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("True language")
    ax.set_title(f"Normalised confusion matrix  (held-out 20%, accuracy {metrics['test_accuracy']:.1%})", loc="left")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).outline.set_visible(False)
    fig.savefig(OUT / "confusion_matrix.png")
    plt.close(fig)

    # 3. Per-language F1
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=order, zero_division=0)
    per = pd.DataFrame({"language": order, "precision": p, "recall": r, "f1": f, "support": s}).sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(per["language"], per["f1"], color=[GROUP_COLOR[LANG_GROUP[l]] for l in per["language"]], height=0.68)
    for y, (v, n) in enumerate(zip(per["f1"], per["support"])):
        ax.text(v + 0.003, y, f"{v:.3f}  (n={n})", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(min(0.8, per["f1"].min() - 0.02), 1.06)
    ax.set_xlabel("F1-score on held-out set")
    ax.set_title("Per-language F1-score", loc="left")
    ax.grid(axis="x", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles, GROUP_COLOR.keys(), frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    fig.savefig(OUT / "per_language_f1.png")
    plt.close(fig)

    # 4. Accuracy versus input length
    lengths = x_te.str.split().str.len().to_numpy()
    bins = [(1, 3, "1–3"), (4, 7, "4–7"), (8, 15, "8–15"), (16, 30, "16–30"), (31, 10**6, "31+")]
    labels, accs, ns = [], [], []
    correct = (pred == y_te.to_numpy())
    for lo, hi, lab in bins:
        m = (lengths >= lo) & (lengths <= hi)
        if m.sum():
            labels.append(lab); accs.append(correct[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(labels, accs, marker="o", color=ACCENT, lw=2.2, ms=7)
    for i, (x, a, n) in enumerate(zip(labels, accs, ns)):
        off, ha = ((14, -6), "left") if i == 0 else ((0, 10), "center")
        ax.annotate(f"{a:.1%}\n(n={n})", (x, a), textcoords="offset points", xytext=off, ha=ha, fontsize=8, color=MUTED)
    ax.set_ylim(min(accs) - 0.05, 1.06)
    ax.set_yticks([t for t in np.arange(0.75, 1.001, 0.05) if t >= min(accs) - 0.05])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Input length (words)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy grows with input length", loc="left")
    ax.grid(axis="y", color=LINE, lw=0.6)
    fig.savefig(OUT / "accuracy_by_length.png")
    plt.close(fig)

    metrics["accuracy_by_length"] = dict(zip(labels, [round(a, 4) for a in accs]))
    metrics["per_language"] = per.set_index("language").round(4).to_dict(orient="index")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=int))
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_language"}, indent=2))


if __name__ == "__main__":
    main()
