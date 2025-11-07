from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn



### Base line imputation
# Knn imputation
class KNN:
    def __init__(self, neighbors, df_missing):
        self.neighbors = neighbors
        self.df = df_missing
        self.columns = self.df.columns[2:].tolist()
        self.scaler = StandardScaler()

    def scaler_imputation(self):
        
        scaled_data = self.scaler.fit_transform(self.df[self.columns])
        knn_imputer = KNNImputer(n_neighbors=self.neighbors)
        
        imputed_data_scaled = knn_imputer.fit_transform(scaled_data)
        
        inverse_scaller = self.scaler.inverse_transform(imputed_data_scaled)

        final_knn_imputation = pd.DataFrame(inverse_scaller, columns=self.columns)

        return final_knn_imputation
    
# Transformer Imputation (Discrete)

class TransformerDiscrete(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3):
        super(TransformerDiscrete, self).__init__()

        self.embedding = nn.Embedding(vocab_size,d_model)
        encoding_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoding_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        embeded = self.embedding(x)
        embeded = embeded.permute(1, 0 ,2)
        transformed = self.transformer(embeded)
        transformed = transformed.permute(1, 0 ,2)
        logits = self.fc_out(transformed)
        return logits
    
# Tranformer Model (Continuous)

def scale_and_mask(df, columns):
    scaler = StandardScaler()

    x_true = scaler.fit_transform(df[columns])
    masked = np.random.rand(*x_true.shape) < 0.2
    x_masked = x_true.copy()
    x_masked[masked] = 0

    return x_true, x_masked, masked

class TransformerRegression(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=3):
        super(TransformerRegression, self).__init__()

        self.input = nn.Linear(num_features,d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_features)
    
    def forward(self, x):
        x = self.input(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        logits = self.fc_out(x)
        return logits