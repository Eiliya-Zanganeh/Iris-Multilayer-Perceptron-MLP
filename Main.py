from torch.utils.data import random_split, DataLoader
from Model import Model
from DataSet import DataSet
import torch

# 0 Load Data

data_set = DataSet('Dataset/iris/iris.csv')

# 1 Split Data

train_dataset, validation_dataset, test_dataset = random_split(data_set, (110, 10, 30))

# 2 Set Data Loader

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# 4 initial Model

model = Model()

# 5 Choice Loss & Optimizer Function

loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6 Train Model

for num in range(500):
    for data_batch, label_batch in train_loader:
        optimizer.zero_grad()
        out = model(data_batch)

        loss = loss_function(out, label_batch)

        loss.backward()

        optimizer.step()

    if (num + 1) % 50 == 0:
        for data_batch, label_batch in validation_loader:
            out = model(data_batch)
            predicted = torch.max(out.data, 1)
            result = int(100 * torch.sum(label_batch == predicted[1]) / len(validation_dataset))
            print(f"validation {num + 1} : {result}")

# 7 Test Model
# model.load_state_dict(torch.load('model_weights.pth'))
for data_batch, label_batch in test_loader:
    out = model(data_batch)
    predicted = torch.max(out.data, 1)
    result = int(100 * torch.sum(label_batch == predicted[1]) / len(test_dataset))
    print(f"test : {result}")

# torch.save(model.state_dict(), 'model_weights.pth')
