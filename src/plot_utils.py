import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

# Default paper figure directory: <repo>/figures/paper
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = _REPO_ROOT / "figures" / "paper"


def configure_pdf_savefig(dpi=300):
    """Matplotlib defaults for paper-ready vector PDFs (editable TrueType text)."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure_pdf(fig, stem, out_dir=None, dpi=300, counter=None):
    """
    Save ``fig`` as a vector PDF under ``out_dir`` (default: repo ``figures/paper``).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    stem : str
        Base filename (spaces/slashes sanitized). Extension is always ``.pdf``.
    out_dir : str | Path | None
    dpi : int
        Embedded raster DPI for any non-vector artists (e.g. imshow).
    counter : dict | None
        Optional ``{\"n\": int}`` for zero-padded ordering prefixes (``01_stem.pdf``).

    Returns
    -------
    pathlib.Path
        Absolute path written.
    """
    out = Path(out_dir) if out_dir is not None else DEFAULT_FIGURE_DIR
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", str(stem).strip()).strip("_") or "figure"
    if counter is not None:
        counter["n"] = int(counter.get("n", 0)) + 1
        name = f"{counter['n']:02d}_{safe}.pdf"
    else:
        name = f"{safe}.pdf"
    path = out / name
    configure_pdf_savefig(dpi=dpi)
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "ar_warmup": [
        "Transformer",
    ],
}


def basic_plot(metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        # [USELESS] Noisy when batch-collecting many runs; enable locally if you need run_id trace.
        # print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics
