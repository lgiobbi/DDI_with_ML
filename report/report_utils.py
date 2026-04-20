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
    fig_dir = os.path.join(os.getcwd(), "analysis_results_figs")
    os.makedirs(fig_dir, exist_ok=True)
    
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
    
    summary_df = df.drop(columns=['precision', 'recall', 'fpr', 'tpr'])
    summary_df.to_csv(os.path.join(fig_dir, filename.replace('.png', '.csv')), index=False)
    
    display(Markdown(f"### Summary of Metrics ({title_suffix})\n\n" + summary_df[['short', 'name', 'AUC_mean', 'AUC_std', 'PR_AUC_mean', 'PR_AUC_std']].to_markdown()))


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