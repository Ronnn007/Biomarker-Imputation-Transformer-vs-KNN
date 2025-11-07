import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_knn_performance (normalised_knn):
    mae_series = pd.Series(normalised_knn).sort_values()

    plt.figure(figsize=(10,8))
    mae_series.plot(kind='barh', color='skyblue', edgecolor='black')
    plt.title('KNN Imputation Performance (MAE by Biomarker)', fontsize=14)
    plt.xlabel('Mean absolute Error (MAE)', fontsize=12)
    plt.ylabel('Biomarker', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_transformer_discrete(normalised_mae):

    mae_series = pd.Series(normalised_mae).sort_values()
    plt.figure(figsize=(10, 6))
    mae_series.plot(kind='barh', color='salmon', edgecolor='black')
    plt.title('Transformer Imputation Performance (Normalized MAE by Biomarker)', fontsize=14)
    plt.xlabel('Relative MAE (normalized by std)', fontsize=12)
    plt.ylabel('Biomarker', fontsize=12)
    plt.xscale('log')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
def plot_transformer_continuous(norm_mae):
    norm_mae_regression = pd.Series(norm_mae).sort_values()
    plt.figure(figsize=(10,6))
    norm_mae_regression.plot(kind='barh', color='coral', edgecolor='black')
    plt.title("Transformer Regression Imputation Performance per Biomarker", fontsize=14)
    plt.xlabel("Relative MAE (normalised by std)", fontsize=12)
    plt.ylabel("Biomarker", fontsize=12)
    plt.xscale('log')
    plt.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_regression_vs_knn(normalised_df, normalised_knn, original_df):
    plt.figure(figsize=(14,7))
    plt.bar(normalised_df.columns, normalised_df.loc['Norm_MAE'], label='Transformer (Regression)', alpha=1, color='red', edgecolor='black', linestyle="--", linewidth=1.4)
    plt.bar(normalised_knn.keys(), [v / original_df[k].std() for k,v in normalised_knn.items()],
            label='KNN (Baseline)', alpha=0.3, color='dimgrey', edgecolor='black', linestyle="--", linewidth=1.4)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.ylabel("Normalized MAE", fontsize=12)
    plt.legend(fontsize=12)
    plt.title("Comparison of Imputation Methods per Biomarker", fontsize=12)
    plt.grid(alpha=0.8)
    plt.show()