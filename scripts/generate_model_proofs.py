import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.lightgbm
from ml.predict import predict_dataframe, load_model_and_encoders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data_output/fastf1_races.parquet")
    parser.add_argument("--output-dir", default="frontend/public/metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_parquet(args.data_path)
    
    # We will evaluate on the 2023 and 2024 seasons as a test set approximation
    test_df = df[df["season"] >= 2023].copy()
    if test_df.empty:
        test_df = df.copy() # fallback
        
    print(f"Test data size: {len(test_df)}")

    # Ensure y_true is present. In our data, label is usually whether they won (finish_position == 1)
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
    
    # Apply a styling theme
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#111827", "figure.facecolor": "#111827", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    cyan = "#00f0ff"
    red = "#ff2a2a"
    
    # 1. ROC Curve
    print("Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=cyan, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=15)
    plt.legend(loc="lower right", facecolor="#1f2937", edgecolor=cyan)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=300, transparent=True)
    plt.close()

    # 2. Precision-Recall Curve
    print("Plotting Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color=cyan, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, pad=15)
    plt.legend(loc="lower left", facecolor="#1f2937", edgecolor=cyan)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pr_curve.png"), dpi=300, transparent=True)
    plt.close()

    # 3. Confusion Matrix (Threshold = 0.5)
    # Since it's highly imbalanced, threshold might need tuning. Let's use 0.5 for now.
    print("Plotting Confusion Matrix...")
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Win", "Win"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    # Style tweaks
    ax.set_title('Confusion Matrix (Threshold=0.5)', fontsize=14, pad=15)
    for text in disp.text_.ravel():
        text.set_color("white")
        text.set_fontsize(14)
        text.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300, transparent=True)
    plt.close()

    # 4. Global Feature Importance (Average SHAP magnitude)
    print("Plotting Feature Importance...")
    shap_dicts = preds_df["shap_values"]
    shap_df = pd.DataFrame(shap_dicts.tolist())
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True).tail(15)
    
    plt.figure(figsize=(10, 8))
    # Horizontal bar chart
    bars = plt.barh(mean_abs_shap.index, mean_abs_shap.values, color=cyan, alpha=0.8)
    plt.xlabel('Mean |SHAP Value| (Impact on Model Output)', fontsize=12)
    plt.title('Global Feature Importance', fontsize=14, pad=15)
    
    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2., f'{width:.3f}', 
                 ha='left', va='center', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_importance.png"), dpi=300, transparent=True)
    plt.close()

    print(f"Proofs generated successfully in {args.output_dir}")

if __name__ == "__main__":
    main()
