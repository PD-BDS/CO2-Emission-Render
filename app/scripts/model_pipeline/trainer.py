import torch
import logging
from torch.utils.data import DataLoader
from scripts.model_pipeline.model_definitions import AttentionLSTMModel, TimeSeriesDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(X_train, y_train, X_val, y_val, input_size, output_window, best_params, model_path):
    model = AttentionLSTMModel(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        output_size=output_window,
        dropout=best_params['dropout']
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=64)

    best_loss = float('inf')
    early_stop = 0

    for epoch in range(30):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch+1}: Validation Loss = {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 5:
            logging.info("⛔ Early stopping triggered.")
            break

    logging.info(f"✅ Model training completed. Best validation loss: {best_loss:.6f}")
    return model
