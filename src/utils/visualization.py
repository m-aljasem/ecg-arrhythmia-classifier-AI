"""Visualization utilities for ECG data and model results"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_samples(ecg_data, sample_idx=0, figsize=(20, 10)):
    """
    Plot ECG signal samples.
    
    Args:
        ecg_data: ECG signal array
        sample_idx: Sample index to plot
        figsize: Figure size
    """
    sample = ecg_data[sample_idx]
    fig, axes = plt.subplots(sample.shape[1], 1, figsize=figsize)
    
    for i in range(sample.shape[1]):
        sns.lineplot(x=np.arange(sample.shape[0]), y=sample[:, i], ax=axes[i])
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, figsize=(16, 4)):
    """
    Plot training history.
    
    Args:
        history: Training history object from model.fit()
        figsize: Figure size
    """
    history_df = pd.DataFrame(history.history)
    sns.relplot(data=history_df, kind='line', height=4, aspect=4)
    plt.show()


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, 
                          fontsize=14):
    """
    Print confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        axes: Matplotlib axes object
        class_label: Label for the class
        class_names: List of class names
        fontsize: Font size for labels
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, 
                        columns=class_names)
    
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), 
                                 rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), 
                                 rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Class - " + class_label)
