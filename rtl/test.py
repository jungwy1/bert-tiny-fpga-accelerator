import math
import torch

data_upper = torch.tensor(-8)
data_lower = torch.tensor(1)

x = torch.bitwise_left_shift(data_upper, 18) + data_lower
acc = (-2) * x
res = acc

data_upper = torch.tensor(1)
data_lower = torch.tensor(-5)

x = torch.bitwise_left_shift(data_upper, 18) + data_lower
acc = (3) * x
res += acc

print(torch.bitwise_right_shift(res, 18))
print(torch.bitwise_right_shift(torch.bitwise_left_shift(res,50),50))


