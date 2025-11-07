import pandas as pd
import numpy as np


# Binning and Tokenization

class CreateBins:
    def __init__(self, df, columns):
        self.binned_df = df.copy()
        self.columns = columns

    def create_bins(self, q=10):
        bin_edges = {}

        for col in self.columns:
            cats, bins = pd.qcut(self.binned_df[col], q, 
                                 labels=False, 
                                 retbins=True,
                                 duplicates='drop')
            self.binned_df[col] = cats.astype(str).radd(f'{col}_bin')
            bin_edges[col] = bins
        
        return bin_edges

    def show_bins (self, rows):
        return self.binned_df[self.columns].head(rows)
    
    def create_unique_tokens(self):
        # Each row is a list of tokens
        unique_tokens = self.binned_df[self.columns].astype(str).agg(list, axis=1)
        return unique_tokens
    
    def show_tokens(self, rows):
        # Preview n rows of tokens
        tokens = self.create_unique_tokens()
        for i in range(min(rows, len(tokens))):
            print(f'Row {i}: {tokens.iloc[i]}')

    
class BiomarkerTokenizer:
    def __init__(self, bin_edges=None):
        self.vocab = {}
        self.id2token = {}
        self.mask_id = None
        self.bin_edges = bin_edges

    def build_vocab(self, unique_tokens):
        all_tokens = set()
        for row in unique_tokens:
            for token in row:
                all_tokens.add(token)

        id = 0

        for token in sorted(all_tokens):
            self.vocab[token] = id
            self.id2token[id] = token
            id += 1

            # Masking
            self.mask_id = id
            self.vocab["[MASK]"] = self.mask_id
            self.id2token[self.mask_id] = "[MASK]"

    def encode(self, unique_tokens):
        # converting list of token sequences into integer sequences

        sequences = []
        for row in unique_tokens:
            seq = []
            for token in row:
                seq.append(self.vocab[token])
            sequences.append(seq)
        
        return sequences

    def decode(self, sequences):
        # convert integers back to tokens
        decoded_sequences = []

        for seq in sequences:
            row_tokens = []
            for token_id in seq:
                row_tokens.append(self.id2token[token_id])
            decoded_sequences.append(row_tokens)
        return decoded_sequences
        

    def token_to_value(self, token):
        # Convert a token back to an approximate numberic value using bin edges

        if token == "[MASK]":
            return np.nan
        
        # e.g. "Glucose_bin3"
        parts = token.split("_bin")
        biomarker = parts[0]
        bin_idx = int(parts[1])
        edges = self.bin_edges[biomarker]
        
        # clip to available bins
        if bin_idx >= len(edges)-1:
            bin_idx = len(edges)-2
        return 0.5 * (edges[bin_idx] + edges[bin_idx+1])
    

def masked_sequence(sequences, mask_id, mask_probaility=0.2):
    masked_sequences = []
    targets = []

    for sequence in sequences:
        masked_seq = []
        target_seq = []

        for token in sequence:
            if np.random.rand() < mask_probaility:
                masked_seq.append(mask_id)
                target_seq.append(token)
            else:
                masked_seq.append(token)
                target_seq.append(-100)
        masked_sequences.append(masked_seq)
        targets.append(target_seq)
    return masked_sequences, targets