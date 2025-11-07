import torch
from torch.utils.data import DataLoader
import sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def knn_evalutation(knn_imputation, df_original, df_missing, columns):
    missing_mask = df_missing[columns].isnull()
    normalized_knn = {}

    for col in columns:
        if missing_mask[col].any():
            original_values = df_original.loc[missing_mask[col], [col]]
            imputed_values = knn_imputation.loc[missing_mask[col], [col]]
            mae = mean_absolute_error(original_values, imputed_values)
            std = df_original[col].std()
            normalized_knn[col] = mae / std

    return normalized_knn

def impute_sequence(model, sequences, mask_id, device="cpu"):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for sequence in sequences:
            masked = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
            output = model(masked)
            preds = torch.argmax(output, dim = -1).squeeze(0).cpu().tolist()

            filled_sequence = []
            for token, prediction in zip(sequence, preds):
                if token == mask_id:
                    filled_sequence.append(prediction)

                else:
                    filled_sequence.append(token)
            
            all_predictions.append(filled_sequence)

    return all_predictions


def evaluate_discreteformer(tokenizer, df, columns, targets, predicted_sequence):

    final_mae = {}
    final_rmse = {}

    for idx, biomarker in enumerate(columns):
        all_true = []
        all_pred = []

        for i in range(len(targets)):
            
            if int(targets[i][idx]) !=-100:
                ture_token = tokenizer.id2token[int(targets[i][idx])]
                pred_token = tokenizer.id2token[int(predicted_sequence[i][idx])]

                true_val = tokenizer.token_to_value(ture_token)
                pred_val = tokenizer.token_to_value(pred_token)

                all_true.append(true_val)
                all_pred.append(pred_val)

        if len(all_true) > 0:

            mae = mean_absolute_error(all_true, all_pred)
            rmse = root_mean_squared_error(all_true, all_pred)

            final_mae[biomarker] = mae
            final_rmse[biomarker] = rmse

    normalised_mae = {}
    normalised_rmse = {}

    for biomarker, mae in final_mae.items():
        
        std = df[biomarker].std()
        normalised_mae[biomarker] = mae /std

    for biomarker, rmse in final_rmse.items():

        std = df[biomarker].std()
        normalised_rmse[biomarker] = rmse / std

    return normalised_mae, normalised_rmse



def evaulate_regressionformer(model, df, dataset):
    model.eval()

    mae_per_biomarker = {}
    rmse_per_biomarker = {}
    norm_mae = {}
    norm_rmse = {}

    true_by_feature = {col: [] for col in df.columns[2:]}
    pred_by_feature = {col: [] for col in df.columns[2:]}

    with torch.no_grad():
        for masked, target, mask in DataLoader(dataset, batch_size=128, shuffle=True):
            masked, target, mask = masked.to(device), target.to(device), mask.to(device)

            output = model(masked)

            for i, col in enumerate(df.columns[2:]):
                masked_indices = mask[:, i]
                if masked_indices.any():
                    true_vals = target[:, i][masked_indices].cpu().numpy()
                    pred_vals = output[:, i][masked_indices].cpu().numpy()

                    true_by_feature[col].extend(true_vals)
                    pred_by_feature[col].extend(pred_vals)
    
    for col in df.columns[2:]:

        true = np.array(true_by_feature[col])
        pred = np.array(pred_by_feature[col])

        if len(true_vals) == 0:
            continue

        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)
        std = df[col].std()
        mae_per_biomarker[col] = mae
        rmse_per_biomarker[col] = rmse
        norm_mae[col] = mae / std
        norm_rmse[col] = rmse / std

    return  mae_per_biomarker, rmse_per_biomarker, norm_mae, norm_rmse   