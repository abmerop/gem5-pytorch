#!/usr/bin/env python3

import torch
print("GPU available!") if torch.cuda.is_available() else print("No GPU available.")

# Seed to ensure same results each run
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

x = torch.rand(5, 3)
y = torch.rand(3, 5)

# Compute on CPU for reference.
z = x @ y
print(f"{z = }")

# Compute on GPU
x = x.to('cuda')
y = y.to('cuda')

z = x @ y

print(f"{z.to('cpu') = }")
