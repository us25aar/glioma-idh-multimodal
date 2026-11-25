"""Diagnostic figures from independent development-fold predictions."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score


def make_validation_figures(output: Path) -> None:
    predictions = pd.read_csv(output / 'validation_oof.csv')
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode, frame in predictions.groupby('mode'):
        fpr, tpr, _ = roc_curve(frame.idh_label, frame.raw_probability)
        auc = roc_auc_score(frame.idh_label, frame.raw_probability)
        axes[0].plot(fpr, tpr, label=f'{mode} ({auc:.3f})')
        observed, predicted = calibration_curve(frame.idh_label, frame.calibrated_probability,
                                                n_bins=5, strategy='quantile')
        axes[1].plot(predicted, observed, marker='o', label=mode)
    for axis in axes:
        axis.plot([0, 1], [0, 1], '--', color='gray', linewidth=.8)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=7, frameon=False)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Outer-fold raw discrimination')
    axes[1].set(xlabel='Mean calibrated probability', ylabel='Observed mutant fraction',
                title='Outer-fold calibration')
    fig.savefig(figures / 'roc_and_calibration.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for axis, (mode, frame) in zip(axes.flat, predictions.groupby('mode')):
        matrix = confusion_matrix(frame.idh_label, frame.prediction, labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=['Wildtype', 'Mutant']).plot(
            ax=axis, colorbar=False, cmap='Blues', values_format='d')
        axis.set_title(mode)
    axes.flat[-1].axis('off')
    fig.savefig(figures / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
