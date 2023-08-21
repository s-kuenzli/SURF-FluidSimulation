import torch
import numpy as np
import torch.nn as nn
from Models.Base import MLP, GNN

NODE_NORMAL = 0
NODE_SYMMETRY = 1
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class MeshGraphNetDiffLoss(nn.Module):
    def __init__(self, apply_noise=True, N=15, state_size=3, parameter_size=0, noise_std=2e-2):
        super(MeshGraphNetDiffLoss, self).__init__()

        self.apply_noise = apply_noise
        self.noise_std = noise_std
        self.encoder = Encoder(state_size, parameter_size)
        self.processor = Processor(N)
        self.decoder = MLP(input_size=128, output_size=state_size, layer_norm=False)
        self.normalizer_output = Normalizer(state_size)

    def forward(self, mesh_pos, edges, state, node_type, parameters):

        if self.apply_noise:
            mask = torch.logical_or(node_type[:, 0, :, NODE_NORMAL] == 1, node_type[:, 0, :, NODE_OUTPUT] == 1)
            noise = torch.randn_like(state[:, 0]).to(state[:, 0].device) * self.noise_std
            state[:, 0][mask[:, 0]] = state[:, 0][mask[:, 0]] + noise[mask[:, 0]]

        state_hat, output_hat = [state[:, 0]], []
        target = []

        for t in range(1, state.shape[1]):
            V, E = self.encoder(mesh_pos[:, t - 1], edges[:, t - 1], node_type[:, t - 1], state_hat[-1], parameters[:, t - 1])
            V, E = self.processor(V, E, edges[:, t - 1])

            next_output = self.decoder(V)

            output_denormalized = self.normalizer_output.inverse(next_output)
            next_state = state_hat[-1] + output_denormalized

            mask = (node_type[:, t, :, NODE_INPUT] == 1) | (node_type[:, t, :, NODE_WALL] == 1) | (
                    node_type[:, t, :, NODE_DISABLE] == 1)

            target_t = self.normalizer_output(state[:, t] - state_hat[-1])
            target.append(target_t)
            next_state[mask, :] = state[:, t][mask, :]
            state_hat.append(next_state)
            output_hat.append(next_output)

        state_hat  = torch.stack(state_hat, dim=1)
        output_hat = torch.stack(output_hat, dim=1)
        target     = torch.stack(target, dim=1)

        return state_hat, output_hat, target


class Encoder(nn.Module):
    def __init__(self, state_size, parameter_size=0):
        super(Encoder, self).__init__()

        self.normalize_edges = Normalizer(3)
        self.normalize_nodes = Normalizer(9 + state_size + parameter_size)

        self.fv = MLP(input_size=9 + state_size + parameter_size)
        self.fe = MLP(input_size=3)

    def forward(self, mesh_pos, edges, node_type, state_hat, parameters):
        # Get nodes embeddings
        V = torch.cat([state_hat, node_type, parameters], dim=-1)

        # Get edges attr
        senders   = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))

        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)

        V = self.fv(self.normalize_nodes(V))
        E = self.fe(self.normalize_edges(E))

        return V, E


class Processor(nn.Module):
    def __init__(self, N=15):
        super(Processor, self).__init__()
        self.gnn = nn.ModuleList([])
        for i in range(N):
            self.gnn.append(GNN())

    def forward(self, V, E, edges):
        for i, gn in enumerate(self.gnn):
            edges = edges
            v, e = gn(V, E, edges)
            V = V + v
            E = E + e

        V = V
        E = E
        return V, E


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.delta = MLP(input_size=128, output_size=3, layer_norm=False)

    def forward(self, V, velocity):
        output   = self.delta(V)
        velocity = velocity + output[..., :2]
        pressure = output[..., -1:]
        return velocity, pressure, output


class Normalizer(nn.Module):
    def __init__(self, input_size):
        super(Normalizer, self).__init__()
        self.acc_sum         = nn.Parameter(torch.zeros(input_size, dtype=torch.float64), requires_grad=False)
        self.acc_sum_squared = nn.Parameter(torch.zeros(input_size, dtype=torch.float64), requires_grad=False)
        self.mean            = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.std             = nn.Parameter(torch.ones(input_size),  requires_grad=False)
        self.max_acc         = 1e7
        self.count           = 0
        self.input_size      = input_size

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])
        if self.training is True:
            if self.count < self.max_acc:
                self.count += 1 
                self.acc_sum.data = self.acc_sum.data + torch.mean(x, dim=(0, 1), dtype=torch.float64)
                self.acc_sum_squared.data = self.acc_sum_squared.data + torch.mean(x ** 2, dim=(0, 1), dtype=torch.float64)
                self.mean.data = (self.acc_sum.data / (self.count + 1e-8)).float()
                self.std.data = torch.sqrt(self.acc_sum_squared.data / (self.count + 1e-8) - self.mean.data ** 2).float()
            if self.input_size == 16:
                # parameters do not have any variation across domain only across simulations
                if (self.count < 1) or (torch.isnan(self.std.data).any()): 
                    self.std.data[-2] = 1.0
                    self.std.data[-1] = 1.0
        
        x = x.reshape(-1, original_shape[-1])
        x = (x - self.mean) / (self.std + 1e-8)
        x = x.reshape(original_shape)

        return x    

    def inverse(self, x):
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        x = x * self.std + self.mean
        return x.reshape(original_shape)


class Projector(nn.Module):
    def __init__(self, neighborhood_size=3):
        super(Projector, self).__init__()

    def forward(self, source_mesh, target_mesh, state):
        distances = torch.cdist(source_mesh.detach(), target_mesh.detach(), p=2)

if __name__ == '__main__':
    model = MeshGraphNetDiffLoss(apply_noise=True, state_size=3, N=10)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)
