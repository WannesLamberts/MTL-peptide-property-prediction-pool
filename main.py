import pandas as pd

import torch

# Create two example tensors
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Append the tensors along the first dimension (rows)
result = torch.cat((tensor1, tensor2), dim=0)

print(result)