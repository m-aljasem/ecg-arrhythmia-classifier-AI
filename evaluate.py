"""
Evaluation script for ECG classification models

Classification of Life-Threatening Arrhythmia ECG Signals Using Deep Learning

Author: Mohamad AlJasem
Website: https://aljasem.eu.org
GitHub: https://github.com/m-aljasem/ecg-arrhythmia-classifier-AI
Contact: mohamad@aljasem.eu.org
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, auc, confusion_matrix
)
from src.data import AugmentedDataGenerator
from config import (
    MODEL01_CHECKPOINT, MODEL02_CHECKPOINT, MODEL03_CHECKPOINT,
    PROCESSED_DATA_FILE, SUPERCLASSES, MODEL_CONFIGS
)

sns.set_style('darkgrid')


def load_test_data():
    """Load test data"""
    with np.load(PROCESSED_DATA_FILE) as data:
        return {
            'X_test': data['X_test'].astype(float),
            'Y_test': data['Y_test'].astype(float),
            'Z_test': data['Z_test'].astype(int),
        }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive metrics for multi-label classification
    
    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_pred: Predicted labels (n_samples, n_classes)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        metrics['auc_roc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
    except ValueError:
        metrics['auc_roc_macro'] = 0.0
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(SUPERCLASSES):
        per_class_metrics[class_name] = {
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }
        try:
            per_class_metrics[class_name]['auc_roc'] = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        except ValueError:
            per_class_metrics[class_name]['auc_roc'] = 0.0
    
    metrics['per_class'] = per_class_metrics
    
    return metrics


def plot_comparison_metrics(all_metrics, output_dir='outputs'):
    """
    Create comprehensive comparison visualizations
    
    Args:
        all_metrics: dict of model_name -> metrics
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    models = list(all_metrics.keys())
    
    # Overall metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall metrics
    overall_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'auc_roc_macro']
    metric_names = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'AUC-ROC (Macro)']
    
    for idx, (metric, name) in enumerate(zip(overall_metrics, metric_names)):
        row = idx // 3
        col = idx % 3
        values = [all_metrics[model][metric] for model in models]
        axes[row, col].bar(models, values, color=['#3498db', '#e74c3c', '#2ecc71'][:len(models)])
        axes[row, col].set_title(name, fontweight='bold')
        axes[row, col].set_ylim([0, 1])
        axes[row, col].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[row, col].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Remove the last subplot (2, 2) since we only have 5 metrics
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/overall_metrics_comparison.png")
    plt.show()
    
    # Per-class AUC-ROC comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SUPERCLASSES))
    width = 0.25
    
    for idx, model in enumerate(models):
        auc_values = [all_metrics[model]['per_class'][cls]['auc_roc'] for cls in SUPERCLASSES]
        offset = (idx - len(models)//2) * width
        ax.bar(x + offset, auc_values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Diagnostic Class', fontweight='bold')
    ax.set_ylabel('AUC-ROC Score', fontweight='bold')
    ax.set_title('Per-Class AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SUPERCLASSES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_auc_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/per_class_auc_comparison.png")
    plt.show()


def plot_roc_curves(y_true, all_predictions, output_dir='outputs'):
    """
    Plot ROC curves for all models and classes
    
    Args:
        y_true: Ground truth labels
        all_predictions: dict of model_name -> predictions_proba
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ROC Curves by Diagnostic Class', fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, class_name in enumerate(SUPERCLASSES):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for model_idx, (model_name, y_pred_proba) in enumerate(all_predictions.items()):
            fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred_proba[:, idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[model_idx], lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_name}', fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot if we have 5 classes
    if len(SUPERCLASSES) == 5:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/roc_curves.png")
    plt.show()


def plot_confusion_matrices(y_true, all_predictions, output_dir='outputs'):
    """
    Plot confusion matrices for all models
    
    Args:
        y_true: Ground truth labels
        all_predictions: dict of model_name -> predictions (binary)
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, y_pred in all_predictions.items():
        fig, axes = plt.subplots(1, 5, figsize=(20, 3))
        fig.suptitle(f'Confusion Matrices - {model_name}', fontsize=14, fontweight='bold')
        
        for idx, class_name in enumerate(SUPERCLASSES):
            cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['N', 'Y'], yticklabels=['N', 'Y'],
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(class_name, fontweight='bold')
            axes[idx].set_ylabel('True')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{model_name}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/confusion_matrix_{model_name}.png")
        plt.show()


def create_comparison_table(all_metrics, output_dir='outputs'):
    """
    Create and save comparison tables
    
    Args:
        all_metrics: dict of model_name -> metrics
        output_dir: directory to save tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall metrics table
    overall_data = []
    for model_name, metrics in all_metrics.items():
        overall_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
            'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
            'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
            'AUC-ROC (Macro)': f"{metrics['auc_roc_macro']:.4f}",
        })
    
    overall_df = pd.DataFrame(overall_data)
    print("\n=== Overall Performance Comparison ===")
    print(overall_df.to_string(index=False))
    overall_df.to_csv(f'{output_dir}/overall_metrics.csv', index=False)
    print(f"Saved: {output_dir}/overall_metrics.csv")
    
    # Per-class metrics table
    per_class_data = []
    for model_name, metrics in all_metrics.items():
        for class_name, class_metrics in metrics['per_class'].items():
            per_class_data.append({
                'Model': model_name,
                'Class': class_name,
                'F1': f"{class_metrics['f1']:.4f}",
                'Precision': f"{class_metrics['precision']:.4f}",
                'Recall': f"{class_metrics['recall']:.4f}",
                'AUC-ROC': f"{class_metrics['auc_roc']:.4f}",
            })
    
    per_class_df = pd.DataFrame(per_class_data)
    print("\n=== Per-Class Performance ===")
    print(per_class_df.to_string(index=False))
    per_class_df.to_csv(f'{output_dir}/per_class_metrics.csv', index=False)
    print(f"Saved: {output_dir}/per_class_metrics.csv")


def evaluate_model(model_name: str, model_path: str, test_data: dict):
    """Evaluate a single model and return predictions"""
    print(f"\n=== Evaluating {model_name} ===")
    
    model = keras.models.load_model(model_path)
    
    # Generate predictions based on model type
    if model_name == 'model01':
        y_pred_proba = model.predict(test_data['X_test'])
    elif model_name in ['model02', 'model03']:
        if model_name == 'model03':
            # Truncate Y_test to 800 samples for model03
            y_pred_proba = model.predict([
                test_data['X_test'],
                test_data['Y_test'][:, :800, :]
            ])
        else:
            y_pred_proba = model.predict([
                test_data['X_test'],
                test_data['Y_test']
            ])
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return {
        'predictions': y_pred,
        'predictions_proba': y_pred_proba
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ECG classification models')
    parser.add_argument('--model', type=str, choices=['model01', 'model02', 'model03', 'all'],
                       default='all', help='Which model to evaluate')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save evaluation outputs')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots (only show metrics)')
    
    args = parser.parse_args()
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    Z_test = test_data['Z_test']
    
    # Evaluate models
    model_configs = []
    if args.model in ['model01', 'all']:
        model_configs.append(('model01', MODEL01_CHECKPOINT))
    if args.model in ['model02', 'all']:
        model_configs.append(('model02', MODEL02_CHECKPOINT))
    if args.model in ['model03', 'all']:
        model_configs.append(('model03', MODEL03_CHECKPOINT))
    
    all_metrics = {}
    all_predictions = {}
    all_predictions_proba = {}
    
    for model_name, model_path in model_configs:
        results = evaluate_model(model_name, model_path, test_data)
        metrics = calculate_metrics(
            Z_test,
            results['predictions'],
            results['predictions_proba']
        )
        all_metrics[model_name] = metrics
        all_predictions[model_name] = results['predictions']
        all_predictions_proba[model_name] = results['predictions_proba']
    
    # Create comparison tables
    create_comparison_table(all_metrics, args.output_dir)
    
    # Generate plots if not skipped
    if not args.skip_plots and len(model_configs) > 0:
        print("\nGenerating visualizations...")
        
        if len(model_configs) > 1:
            plot_comparison_metrics(all_metrics, args.output_dir)
        
        plot_roc_curves(Z_test, all_predictions_proba, args.output_dir)
        plot_confusion_matrices(Z_test, all_predictions, args.output_dir)
    
    print(f"\n✅ Evaluation complete! All outputs saved to '{args.output_dir}/' directory")


if __name__ == '__main__':
    main()
