import torch


def orthogonal_regularization(net, reg_coef=1e-3, device=torch.device('cuda')):
    assert isinstance(net, torch.nn.Module)

    reg = 0
    for layer in net.modules():
        if isinstance(layer, torch.nn.Linear):
            prod = torch.matmul(torch.transpose(layer.weight, 0, 1), layer.weight)
            reg += torch.sum(torch.square(prod * (1 - torch.eye(prod.shape[0]).to(device))))

    return reg * reg_coef


def soft_update(net, target_net, tau=0.005):
    assert isinstance(net, torch.nn.Module) and isinstance(target_net, torch.nn.Module)

    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
