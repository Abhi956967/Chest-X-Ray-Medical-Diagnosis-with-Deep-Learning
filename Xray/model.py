# xray/model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example: simple linear layer (you can modify as per your trained model)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.fc(x)
