import torch
from torch import nn
from torch.utils.data import DataLoader
from custom_transformer import CustomTransformer  
from dataset import load_train_data, run_test_predictions

def build_model():
    model = CustomTransformer(d_model=3, n_heads=1, d_ff=12)  
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs=350):
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features, batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def evaluate_model(model, test_loader):
    from sklearn.metrics import mean_squared_error
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_predictions = model(batch_features, batch_targets)
            predictions.extend(batch_predictions.cpu().numpy())
            actuals.extend(batch_targets.cpu().numpy())
    mse = mean_squared_error(actuals, predictions)
    print(f'Mean Squared Error: {mse}')
    return mse

def main():
    train_loader, test_loader, scaler, selected_sites = load_train_data()  # 数据加载来自dataset.py
    model = build_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, test_loader)
    run_test_predictions(model, scaler, selected_sites)

if __name__ == "__main__":
    main()

