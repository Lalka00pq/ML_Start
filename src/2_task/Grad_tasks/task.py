import torch
w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)

function = torch.log(torch.log(w+7)).prod()
function.backward()
print(w.grad)
