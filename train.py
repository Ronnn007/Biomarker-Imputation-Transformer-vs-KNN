import torch
import torch.nn as nn
from tqdm import tqdm


def train_transformer_discrete(model, dataloader, vocab, device, epochs, lr=1e-4):
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        
        for masked, target in tqdm(dataloader, disable=(epoch % 10 != 0)):
            masked = masked.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(masked)

            loss = criterion(output.view(-1, len(vocab)), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model

def train_regression_transformer ( model, dataloader, device, epochs, lr=1e-4):
    mse_criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for masked, target, mask in tqdm(dataloader, disable=(epoch % 10 != 0)):
            masked, target, mask = masked.to(device), target.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(masked)

            loss = mse_criterion(output, target)
            loss = loss[mask].mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")