import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
from pathlib import Path

print(f"torch version: {torch.__version__}")
print("-" * 80)

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")
print("-" * 80)

# prepare data
print("# prepare data: ")
weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias
print(f"X_shape: {X.shape}")
print(f"y_shape: {y.shape}")

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split].to(device), y[:train_split].to(device)
X_test, y_test = X[train_split:].to(device), y[train_split:].to(device)
print(f"X_train_shape: {X_train.shape}")
print(f"y_train_shape: {y_train.shape}")
print(f"X_test_shape: {X_test.shape}")
print(f"y_test_shape: {y_test.shape}")
print("-" * 80)


def plot_preds(X_train, y_train, X_test, y_test, preds=None):
    plt.figure(figsize=(10, 7))
    # plot train data in blue
    plt.scatter(X_train, y_train, c="b", s=4, label="train data")
    # plot test data in green
    plt.scatter(X_test, y_test, c="g", s=4, label="test data")

    if preds is not None:
        plt.scatter(X_test, preds, c="r", s=4, label="predictions")

    # show the legend
    plt.legend(prop={"size": 14})
    plt.show()


plot_preds(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu())

print("# build model: ")


# linear regression model
# class LR(nn.Module):
#     def __init__(self):
#         super(LR, self).__init__()
#         self.weight = nn.Parameter(
#             torch.randn(1, dtype=torch.float), requires_grad=True
#         )
#         self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
#
#     def forward(self, x):
#         return self.weight * x + self.bias
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)


print("-" * 80)

torch.manual_seed(42)
print("# check the contents of model: ")
model = LR().to(device)
pprint(list(model.parameters()))
pprint(model.state_dict())
pprint(next(model.parameters()).device)
print("-" * 80)

print("# make predictions with model: ")
# in older PyTorch code you might also see torch.no_grad()
with torch.inference_mode():
    y_pred = model(X_test)
print(f"y_pred_shape: {y_pred.shape}")
print(f"y_test_shape: {y_test.shape}")
print(f"y_pred: {y_pred}")
print(f"y_test: {y_test}")
print("-" * 80)

plot_preds(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), preds=y_pred.cpu())


# train model
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

train_loss_values = []
test_loss_values = []
count_epoch = []

print("train and test begin!")
for epoch in range(epochs):
    # train
    model.train()

    # forward pass
    y_train_pred = model(X_train)
    # calculate loss
    train_loss = criterion(y_train_pred, y_train)
    # zero grad
    optimizer.zero_grad()
    # loss backward
    train_loss.backward()
    # optimizer step
    optimizer.step()

    # test
    model.eval()

    with torch.inference_mode():
        # forward pass
        y_test_pred = model(X_test)
        # calculate loss
        test_loss = criterion(y_test_pred, y_test)

    if epoch % 10 == 0:
        count_epoch.append(epoch)
        train_loss_values.append(train_loss.detach().cpu().numpy())
        test_loss_values.append(test_loss.detach().cpu().numpy())
        print(
            f"epoch: [{epoch}/{epochs}] | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}"
        )
print("train and test end!")
print("-" * 80)

# plot the loss curve
print("# plot the loss curve: ")
plt.plot(count_epoch, train_loss_values, label="train loss")
plt.plot(count_epoch, test_loss_values, label="test loss")
plt.title("train and test loss curves")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend()
plt.show()
print("-" * 80)

# check the model learned parameters
print("# the model learned weight and bias: ")
pprint(model.parameters())
print("-" * 80)

# make predictions with trained model
print("# make predictions with trained model: ")
with torch.inference_mode():
    model.eval()
    y_pred = model(X_test)

plot_preds(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), preds=y_pred.cpu())
print("-" * 80)

# save and load model
print("# save model: ")
MODEL_PATH = Path("../checkpoints")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "02_pytorch_workflow_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"save model to {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("# load saved model: ")
loaded_model = LR()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

print("# use loaded model to make predictions: ")
with torch.inference_mode():
    loaded_y_pred = loaded_model(X_test.cpu())

print(loaded_y_pred == y_pred.cpu())
plot_preds(
    X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), preds=loaded_y_pred
)
