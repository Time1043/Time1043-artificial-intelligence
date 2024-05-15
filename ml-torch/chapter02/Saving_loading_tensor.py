import torch

a = torch.rand(6)
b = torch.rand(6)
c = torch.rand(6)
print([a, b, c])
torch.save([a, b, c], "output/tensor_abc")
abc_list = torch.load("output/tensor_abc")
print(abc_list)

tensor_dict = {'a': a, 'b': b, 'c': c}
print(tensor_dict)
torch.save(tensor_dict, "output/tensor_dict_abc")
abc_dict = torch.load("output/tensor_dict_abc")
print(abc_dict)
