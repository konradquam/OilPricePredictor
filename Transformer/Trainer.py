import torch
from torch import nn
from OilPricePredictor.src import DataProcessor as DP


eval_interval = None
eval_iters = None
model = None
train_data = None
val_data = None
batch_size = None
MSE = nn.MSELoss()


@torch.no_grad
def estimate_loss():
    model.eval()
    train_loss = None
    val_loss = None
    for split in [train_data, val_data]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = DP.get_batch(split)
            y_hat = model(x)
            losses[i] = MSE(y_hat, y).item()
        if split == train_data:
            train_loss = losses.mean()
        else:
            val_loss = losses.mean()
    model.train()
    return train_loss, val_loss


def train_model(data, learning_rate, iters):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)

    for i in range(iters):
        x, y = DP.get_batch(data)
        y_hat = model(x)

        loss = MSE(y_hat, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0 or i == iters - 1:
            train_loss, val_loss = estimate_loss()
            print(f"step {i}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            torch.save(model, '/./content/drive/MyDrive/Colab Notebooks/OPECModel.pt')
            torch.save(model.state_dict(), '/./content/drive/MyDrive/Colab Notebooks/OPECModel.pt')




