from torch.nn import Module, Linear, functional


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = Linear(4, 10)
        self.fc2 = Linear(10, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.fc2(x)
        return x
