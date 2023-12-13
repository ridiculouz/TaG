import torch.nn as nn
import torch

class RelGCNConv(nn.Module):
    def __init__(self, in_features, out_features, num_rel):
        super(RelGCNConv, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(in_features, out_features, bias=True) for i in range(num_rel)])
        self.tanh = nn.Tanh()


    def forward(self, x: torch.Tensor, adjacency_list):
        x_list = []
        for i, adjacency_hat in enumerate(adjacency_list):
            x_l = self.linears[i](x)
            # x_d = torch.sparse.mm(adjacency_hat, x_l)
            x_d = torch.mm(adjacency_hat, x_l)
            x_list.append(x_d.unsqueeze(0))
        x = self.tanh(torch.sum(torch.cat(x_list, dim=0), dim=0))   # residual connection?
        return x

class RelGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_rel, num_layer):
        super().__init__()
        layers = []
        if num_layer == 1:
            layers.append(RelGCNConv(in_features, out_features, num_rel))
        else:
            for i in range(num_layer):
                if i == 0:
                    layers.append(RelGCNConv(in_features, hidden_features, num_rel))
                elif i == num_layer - 1:
                    layers.append(RelGCNConv(hidden_features, out_features, num_rel))
                else:
                    layers.append(RelGCNConv(hidden_features, hidden_features, num_rel))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adjacency_list):
        for layer in self.layers:
            x = layer(x, adjacency_list)
        return x