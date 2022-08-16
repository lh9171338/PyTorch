"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
numpy
"""
import torch
import numpy as np

# details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations

# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print('\nnumpy array:\n', np_data)
print('torch tensor:\n', torch_data)
print('tensor to array:\n', tensor2array)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print('\nabs')
print('numpy:\n', np.abs(data))
print('torch:\n', torch.abs(tensor))

# sin
print('\nsin')
print('numpy:\n', np.sin(data))
print('torch:\n', torch.sin(tensor))

# mean
print('\nmean')
print('numpy:\n', np.mean(data))
print('torch:\n', torch.mean(tensor))

# matrix multiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print('\nmatrix multiplication (matmul)')
print('numpy:\n', np.matmul(data, data))
print('torch:\n', torch.mm(tensor, tensor))

# # incorrect method
# data = np.array(data)
# print('\nmatrix multiplication (dot)')
# print('numpy:\n', data.dot(data))
# print('torch:\n', tensor.dot(tensor))
