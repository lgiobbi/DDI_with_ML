import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display, clear_output

import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets

# Centralized output directory for this module's generated files
OUTPUT_SUBDIR = "report_outputs"


def get_output_dir(subfolder: str | None = None) -> str:
    """Return and create an output directory inside the report folder.

    If `subfolder` is provided, it will be created under the main output folder.
    """
    base = os.path.join(os.path.dirname(__file__), OUTPUT_SUBDIR)
    if subfolder:
        base = os.path.join(base, subfolder)
    os.makedirs(base, exist_ok=True)
    return base


def get_base_config():
    """Returns a fresh base configuration for experiments to prevent cross-contamination."""
    from ddi_graph_neural_network.config import Config

    config = Config()
    config.training.seed = 42
    config.graph.seed_graph_sampling = 42
    config.graph.current_graph = "DrugBank_CRESCENDDI"
    config.run.take_negative_samples = True
    config.training.repetitions = 1
    config.graph.feature = "DESC_GPT"
    return config


def get_feature_experiment_config():
    """Returns the configuration for the feature embeddings experiment."""
    from ddi_graph_neural_network.config import LossType

    config = get_base_config()
    config.run.loss_type = LossType.WeightedBCEWithLogitsLoss
    config.run.pos_loss_multiplier = 0.5
    config.run.upsample_negative_labels = False
    config.run.use_only_sampled_negatives_in_train = False
    return config


def compute_node_metrics(test_data, test_scores, reversed_node_id_map):
    """
    Computes threshold, mapped predictions, classification mapping and aggregates metrics.
    """
    threshold = find_balanced_threshold(test_scores)
    gt, pred = test_data.edge_label.cpu().numpy().astype(int), (test_scores >= threshold).astype(int)

    src_idx, tgt_idx = test_data.edge_label_index.cpu().numpy()
    sources, targets = np.vectorize(reversed_node_id_map.get)(src_idx), np.vectorize(reversed_node_id_map.get)(tgt_idx)

    pred_classes = np.where(gt == 1, np.where(pred == 1, "TP", "FN"), np.where(pred == 1, "FP", "TN"))

    cols = {
        "source": np.concatenate([sources, targets]),
        "target": np.concatenate([targets, sources]),
        "pred_class": np.tile(pred_classes, 2),
    }
    edges_df = pd.DataFrame(cols)

    node_info_trained = (
        pd.get_dummies(edges_df["pred_class"]).groupby(edges_df["source"]).sum()[["TP", "TN", "FP", "FN"]]
    )
    c = node_info_trained.sum(axis=1)

    node_info_trained = node_info_trained.assign(
        count=c,
        perc_missclassified=(node_info_trained["FP"] + node_info_trained["FN"]) / c * 100,
        perc_neg_gt=(node_info_trained["FP"] + node_info_trained["TN"]) / c * 100,
        balanced_error_rate=0.5
        * (
            node_info_trained["FP"] / (node_info_trained["FP"] + node_info_trained["TN"] + 1e-10)
            + node_info_trained["FN"] / (node_info_trained["TP"] + node_info_trained["FN"] + 1e-10)
        ),
    )
    return threshold, node_info_trained


def compile_embeddings(model, data, reversed_node_id_map, node_info_trained):
    """
    Extracts embeddings, loads ATC classes, and merges everything into a DataFrame.
    """
    import torch
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(data.x, data.edge_index).cpu().numpy()

    tsne = TSNE(n_components=2)
    latent_tsne = pd.DataFrame(tsne.fit_transform(node_embeddings), columns=["TSNE-1_latent", "TSNE-2_latent"])
    latent_tsne.index = pd.Series(latent_tsne.index).map(
        lambda x: reversed_node_id_map.get(int(x), f"unknown_{int(x)}")
    )

    original_df = pd.read_csv("/data/giobbi/embeddings/DESC_GPT.csv", sep="\t", index_col=0)
    original_df.set_index(original_df.columns[0], inplace=True)

    original_tsne = pd.DataFrame(
        tsne.fit_transform(original_df.select_dtypes("number").dropna()),
        index=original_df.index,
        columns=["TSNE-1_input", "TSNE-2_input"],
    )

    atc_features = pd.read_csv(
        "/data/giobbi/embeddings/not_aligned_with_model/drug_description_enriched_atc.csv", sep="\t", index_col=0
    )
    atc_features["atc_class_lvl_1"] = atc_features["atc_class_lvl_1"].str.split(": ").str[-1]

    embedding = (
        original_tsne.join(latent_tsne, how="inner")
        .join(atc_features.set_index("Drug ID"), how="left")
        .join(
            node_info_trained[["perc_missclassified", "balanced_error_rate", "count", "perc_neg_gt"]].rename(
                columns={"count": "edge_count"}
            ),
            how="left",
        )
    )

    embedding["pharma_class"], _ = pd.factorize(embedding["atc_class_lvl_1"].fillna("Unknown"))
    return embedding


def plot_pharma_class_error_rates(embedding):
    """
    Computes summary statistics and plots the distribution of error rates by pharma class.
    """
    df = (
        embedding.dropna(subset=["perc_missclassified"])
        .reset_index(drop=True)
        .assign(class_=lambda x: x["atc_class_lvl_1"].fillna("Unknown"))
    )

    summary_df = (
        df.groupby("class_")["perc_missclassified"]
        .agg(n="count", median="median", IQR=lambda x: x.quantile(0.75) - x.quantile(0.25), mean="mean", std="std")
        .dropna()
        .reset_index()
        .rename(columns={"class_": "class"})
        .sort_values("n", ascending=False)
    )

    display(summary_df.head(20))
    summary_df.to_csv(os.path.join(get_output_dir(), "pharma_class_summary_all.csv"), index=False)

    sns.set_style("whitegrid")
    order = summary_df["class"].tolist()
    edge_counts = df.groupby("class_")["edge_count"].mean().reindex(order)

    norm = plt.Normalize(edge_counts.min(), edge_counts.max())
    cmap = plt.cm.Reds
    palette = {c: cmap(norm(v)) for c, v in edge_counts.items()}

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.boxplot(
        data=df[df["class_"].isin(order)],
        x="class_",
        y="perc_missclassified",
        order=order,
        width=0.6,
        showcaps=True,
        showfliers=False,
        palette=palette,
        showmeans=True,
        meanline=True,
        ax=ax,
        medianprops={"linewidth": 1.5, "color": "black"},
        whiskerprops={"linewidth": 1.2, "color": "black"},
        capprops={"linewidth": 1.2, "color": "black"},
    )

    ax.tick_params(axis="x", rotation=90)
    ax.set(
        title="Mean Error Rate by Pharma Class (colored by edge count)", xlabel="", ylabel="Missclassified Percentage"
    )

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02).set_label(
        "Mean Edge Count per Class", fontsize=11
    )

    plt.tight_layout()
    plt.savefig(os.path.join(get_output_dir(), "pharma_class_all_categories.png"), dpi=200, bbox_inches="tight")
    plt.show()


def plot_experiment_results(results, title_suffix="Setting", filename="report_grid.png"):
    """
    Plots a 2x2 grid representing experiment metrics (ROC Curves, PR Curves, AUC, and PR AUC bars).
    """
    df = pd.DataFrame([{
        "short": f"S{i+1}", "name": r["name"], **r["metrics"], 
        "precision": r["precision"], "recall": r["recall"], "fpr": r["fpr"], "tpr": r["tpr"]
    } for i, r in enumerate(results)]).fillna(0)
    
    labels = df['short'].tolist()
    indices = np.arange(len(labels))
    fig_dir = get_output_dir("analysis_results_figs")
    
    def get_ylim(means, errs, min_range=0.03):
        val_min, val_max = np.min(means - errs), np.max(means + errs)
        margin = max(0.01, 0.06 * (val_max - val_min))
        low, high = max(0.0, val_min - margin), min(1.0, val_max + margin)
        return (max(0.0, high - min_range), high) if high - low < min_range else (low, high)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    (ax_roc_curve, ax_roc_bar), (ax_pr_curve, ax_pr_bar) = axes

    for _, r in df.iterrows():
        ax_roc_curve.plot(r['fpr'], r['tpr'], label=r['short'], linewidth=2)
        ax_pr_curve.plot(r['recall'], r['precision'], label=r['short'], linewidth=2)

    ax_roc_curve.plot([0,1],[0,1],'k--', alpha=0.25)
    ax_roc_curve.set(title=f'ROC Curves ({title_suffix})', xlabel='FPR', ylabel='TPR')
    ax_roc_curve.legend(title='Legend', fontsize='small')

    ax_pr_curve.set(title=f'PR Curves ({title_suffix})', xlabel='Recall', ylabel='Precision')
    ax_pr_curve.legend(title='Legend', fontsize='small', loc='lower left')

    for ax, data_key, color in zip([ax_roc_bar, ax_pr_bar], ['AUC', 'PR_AUC'], ['Blues', 'Oranges']):
        means, errs = df[f'{data_key}_mean'].values, df[f'{data_key}_std'].values
        bars = ax.bar(indices, means, yerr=errs, capsize=6, color=sns.color_palette(color, n_colors=len(labels)))
        ax.set_xticks(indices)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set(ylim=get_ylim(means, errs), title=f'{data_key} per {title_suffix}')
        ax.grid(axis='y', alpha=0.6)
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.002, f"{h:.3f}", ha='center', va='bottom', fontsize=9)

    report_path = os.path.join(fig_dir, filename)
    fig.savefig(report_path, dpi=180)
    plt.show()
    
    summary_df = df.drop(columns=['precision', 'recall', 'fpr', 'tpr']).reset_index(drop=True)
    summary_df.to_csv(os.path.join(fig_dir, filename.replace('.png', '.csv')), index=False)
    
    display(Markdown(f"### Summary of Metrics ({title_suffix})\n\n" + summary_df[['short', 'name', 'AUC_mean', 'AUC_std', 'PR_AUC_mean', 'PR_AUC_std']].to_markdown(index=False)))


def find_balanced_threshold(y_scores):
    """
    Returns median float value from y_scores array ensuring balanced thresholding
    """
    return float(np.median(np.asarray(y_scores).ravel()))


def render_interactive_visualization(embedding, test_data, test_scores, threshold, reversed_node_id_map):
    """
    Spawns interactive dropdown and T-SNE plots for exploring edge maps 
    """
    all_drug_ids = sorted(embedding.index.tolist())

    drug_dropdown = widgets.Dropdown(
        options=[("Show all edges", None)] + [(d, d) for d in all_drug_ids],
        value=None, description="Drug ID:", layout=widgets.Layout(width='400px')
    )
    tsne_toggle = widgets.ToggleButtons(options=['input', 'latent'], description='t-SNE space:')

    edge_label_np = test_data.edge_label.cpu().numpy()
    predicted_labels_np = (test_scores >= threshold).astype(int)
    edge_pairs_np = test_data.edge_label_index.cpu().t().numpy()

    def create_visualization(filter_drug_id, tsne_setting_selection):
        def format_description(desc, max_length=60):
            return "<br>".join(re.findall(".{1,%d}" % max_length, desc)) if pd.notnull(desc) else ""

        tsne_suffix = f"_{tsne_setting_selection}" if tsne_setting_selection else ""
        embedding["Description_br"] = embedding["Discription"].apply(format_description)
        embedding["atc_class_lvl_1"] = embedding["atc_class_lvl_1"].fillna("Unknown")

        classes = sorted(embedding["atc_class_lvl_1"].astype(str).unique())
        unique_palette = list(dict.fromkeys(px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3))
        color_map = {cls: unique_palette[i % len(unique_palette)] for i, cls in enumerate(classes)}

        fig = go.Figure()
        for cls in classes:
            df_cls = embedding[embedding["atc_class_lvl_1"].astype(str) == cls]
            fig.add_trace(go.Scattergl(
                x=df_cls[f"TSNE-1{tsne_suffix}"], y=df_cls[f"TSNE-2{tsne_suffix}"],
                mode="markers", marker=dict(size=10, color=color_map[cls], opacity=0.8, line=dict(width=0.5, color="white")),
                text=df_cls.index.astype(str),
                customdata=df_cls[["Drug Name", "atc_class_lvl_1", "Description_br"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>Drug ID: %{text}<br>ATC Class: <b>%{customdata[1]}</b><br>t-SNE-1: %{x:.2f} | t-SNE-2: %{y:.2f}<br>Description: %{customdata[2]}<extra></extra>",
                name=cls
            ))

        def get_drug_id(n): return reversed_node_id_map.get(int(n))
        def get_emb_key(n):
            for k in (n, str(n), int(n) if str(n).isdigit() else None, get_drug_id(n)):
                if k in embedding.index: return k
            return None

        is_cls = {
            "TP": (edge_label_np == 1) & (predicted_labels_np == 1),
            "FN": (edge_label_np == 1) & (predicted_labels_np == 0),
            "TN": (edge_label_np == 0) & (predicted_labels_np == 0),
            "FP": (edge_label_np == 0) & (predicted_labels_np == 1)
        }

        filter_mask = np.ones(len(edge_pairs_np), dtype=bool)
        if filter_drug_id:
            filter_mask = np.array([filter_drug_id in (get_drug_id(u), get_drug_id(v)) for u, v in edge_pairs_np])

        def add_lines(mask, color, name, dash="solid"):
            xs, ys = [], []
            for (u, v), k, p in zip(edge_pairs_np, mask, filter_mask):
                if k and p:
                    ku, kv = get_emb_key(u), get_emb_key(v)
                    if ku and kv:
                        x0, y0 = embedding.loc[ku, [f"TSNE-1{tsne_suffix}", f"TSNE-2{tsne_suffix}"]]
                        x1, y1 = embedding.loc[kv, [f"TSNE-1{tsne_suffix}", f"TSNE-2{tsne_suffix}"]]
                        xs.extend([x0, x1, None])
                        ys.extend([y0, y1, None])
            if xs: fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.5, dash=dash), name=name, hoverinfo="skip"))

        add_lines(is_cls["TP"], "rgba(0,180,0,0.6)", "TP (gt=1, pred=1)")
        add_lines(is_cls["FN"], "rgba(160,180,0,0.6)", "FN (gt=1, pred=0)", "dash")
        add_lines(is_cls["TN"], "rgba(220,20,60,0.6)", "TN (gt=0, pred=0)")
        add_lines(is_cls["FP"], "rgba(220,120,90,0.6)", "FP (gt=0, pred=1)", "dash")

        counts = {k: np.sum(v & filter_mask) for k, v in is_cls.items()}
        subtitle = f"Filter: {filter_drug_id or 'All edges'} | TP={counts['TP']}, TN={counts['TN']}, FP={counts['FP']}, FN={counts['FN']}<br>Edges represent test set interactions colored by confusion class."

        fig.update_layout(
            title=dict(text=f"t-SNE {tsne_setting_selection.capitalize()} Embedding", subtitle=dict(text=subtitle, font=dict(size=14))),
            width=1450, height=1150, plot_bgcolor="rgba(240, 242, 245, 0.5)", paper_bgcolor="white", hovermode="closest",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="white", bordercolor="silver", borderwidth=1),
            xaxis=dict(title="t-SNE 1", showgrid=True), yaxis=dict(title="t-SNE 2", showgrid=True), margin=dict(l=80, r=20, t=140, b=80)
        )
        fig.show()

    def on_filter_change(change):
        clear_output(wait=True)
        display(widgets.HBox([drug_dropdown, tsne_toggle]))
        create_visualization(drug_dropdown.value, tsne_toggle.value)

    drug_dropdown.observe(on_filter_change, names='value')
    tsne_toggle.observe(on_filter_change, names='value')

    display(widgets.HBox([drug_dropdown, tsne_toggle]))
    create_visualization("DB00007", "input")