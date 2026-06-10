import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import mlflow
import mlflow.lightgbm
from ml.predict import predict_dataframe, load_model_and_encoders

def style_axis(ax):
    """Apply premium neon styling to matplotlib axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(color='#334155', linestyle='--', alpha=0.4)
    ax.set_facecolor('#0f172a')

def add_neon_glow(ax, x, y, color, lw=2):
    """Add a glowing effect to a line."""
    ax.plot(x, y, color=color, lw=lw, zorder=5)
    for n in range(1, 4):
        ax.plot(x, y, color=color, lw=lw + (n*3), alpha=0.1, zorder=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data_output/fastf1_races.parquet")
    parser.add_argument("--output-dir", default="frontend/public/metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_parquet(args.data_path)
    
    test_df = df[df["season"] >= 2023].copy()
    if test_df.empty:
        test_df = df.copy()
        
    print(f"Test data size: {len(test_df)}")

    if "finish_position" in test_df.columns:
        y_true = (test_df["finish_position"] == 1).astype(int)
    else:
        print("No target column found. Cannot generate proofs.")
        return

    print("Loading model and encoders...")
    run_id = os.getenv("KRONECTOR_MODEL_RUN_ID")
    if not run_id:
        print("Please set KRONECTOR_MODEL_RUN_ID")
        return
        
    try:
        model, encoders = load_model_and_encoders(run_id)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Generating predictions...")
    preds_df = predict_dataframe(test_df, model, encoders, explain=True)
    y_pred_prob = preds_df["win_probability"]
    
    # Premium Dark Theme
    plt.style.use('dark_background')
    fig_color = "#0f172a"
    cyan = "#00f0ff"
    red = "#ff2a2a"
    text_color = "#f8fafc"
    
    # 1. ROC Curve
    print("Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=fig_color)
    style_axis(ax)
    
    add_neon_glow(ax, fpr, tpr, cyan)
    ax.fill_between(fpr, tpr, alpha=0.1, color=cyan)
    ax.plot([0, 1], [0, 1], color='#475569', lw=2, linestyle='--')
    
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Receiver Operating Characteristic (ROC)', color=text_color, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend([f'AUC = {roc_auc:.3f}'], loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curve.svg"), format="svg", transparent=True, bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall Curve
    print("Plotting Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=fig_color)
    style_axis(ax)
    
    add_neon_glow(ax, recall, precision, cyan)
    ax.fill_between(recall, precision, alpha=0.1, color=cyan)
    
    ax.set_xlabel('Recall', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Precision', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Precision-Recall Curve', color=text_color, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend([f'AUC = {pr_auc:.3f}'], loc="lower left", facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pr_curve.svg"), format="svg", transparent=True, bbox_inches='tight')
    plt.close()

    # 3. Confusion Matrix
    print("Plotting Confusion Matrix...")
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=fig_color)
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.color_palette("dark:#00f0ff", as_cmap=True), 
                cbar=False, ax=ax, annot_kws={"size": 18, "weight": "bold", "color": "white"},
                linewidths=2, linecolor='#0f172a', square=True)
    
    ax.set_facecolor('#0f172a')
    ax.set_xlabel('Predicted Label', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix (Threshold=0.5)', color=text_color, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(['Not Win', 'Win'], color='#94a3b8', fontsize=12)
    ax.set_yticklabels(['Not Win', 'Win'], color='#94a3b8', fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.svg"), format="svg", transparent=True, bbox_inches='tight')
    plt.close()

    # 4. Global Feature Importance
    print("Plotting Feature Importance...")
    shap_dicts = preds_df["shap_values"]
    shap_df = pd.DataFrame(shap_dicts.tolist())
    
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True).tail(12)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=fig_color)
    style_axis(ax)
    ax.grid(False, axis='y') # Remove horizontal grid lines for bars
    
    # Draw bars with gradient-like glowing effect
    y_pos = np.arange(len(mean_abs_shap))
    ax.barh(y_pos, mean_abs_shap.values, color=cyan, alpha=0.8, height=0.6, edgecolor=cyan, linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(x).replace('_', ' ').title() for x in mean_abs_shap.index], color='#cbd5e1', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Mean |SHAP Value| (Impact on Model Output)', color='#cbd5e1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Global Feature Importance', color=text_color, fontsize=16, fontweight='bold', pad=20)
    
    # Add neon values to bars
    for i, v in enumerate(mean_abs_shap.values):
        ax.text(v + (max(mean_abs_shap.values) * 0.02), i, f'{v:.3f}', 
                color=cyan, fontweight='bold', va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_importance.svg"), format="svg", transparent=True, bbox_inches='tight')
    plt.close()

    print(f"Proofs generated successfully in {args.output_dir}")

if __name__ == "__main__":
    main()
