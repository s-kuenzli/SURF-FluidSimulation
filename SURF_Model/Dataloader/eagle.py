import os.path
import random
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
import numpy as np
from tqdm import tqdm

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class EagleDataset(Dataset):
    def __init__(self, data_path, mode="test", window_length=990, apply_onehot=True, with_cells=False,
                 with_cluster=False, n_cluster=20, normalize=False, max_timestep=50):
        """ Eagle dataset
        :param data_path: path to the dataset
        :param window_length: length of the temporal window to sample the simulation
        :param apply_onehot: Encode node type as onehot vector, see global variables to see what is what
        :param with_cells: Default is to return edges as pairs of indices, if True, return also the triangles (cells),
        useful for visualization
        :param with_cluster: Load the clustered indices of the nodes
        :param n_cluster: Number of cluster to use, 0 means no clustering / one node per cluster
        :param normalize: center mean and std of velocity/pressure/temperature field
        """
        super(EagleDataset, self).__init__()
        assert mode in ["train", "test", "valid", "save"]

        self.window_length = window_length
        assert window_length <= 990, "window length must be smaller than 990"

        self.fn = data_path
        assert os.path.exists(self.fn), f"Path {self.fn} does not exist"

        self.apply_onehot = apply_onehot
        self.dataloc = []

        with open(os.path.join(data_path, f"Splits/{mode}.txt"), "r") as f:
            for line in f.readlines():
                self.dataloc.append(os.path.join(self.fn, line.strip()))

        self.with_cells       = with_cells
        self.with_cluster     = with_cluster
        self.n_cluster        = n_cluster
        self.mode             = mode
        self.do_normalization = normalize
        self.max_timestep     = max_timestep
        if normalize:
            pressure_mean, pressure_std, velocity_mean, velocity_std, temperature_mean, temperature_std, parameters_mean, parameters_std = calculate_statistics(self.fn)
            self.pressure_mean    = torch.tensor(pressure_mean).float()
            self.pressure_std     = torch.tensor(pressure_std).float()
            self.velocity_mean    = torch.tensor(velocity_mean).float()
            self.velocity_std     = torch.tensor(velocity_std).float()
            self.temperature_mean = torch.tensor(temperature_mean).float()
            self.temperature_std  = torch.tensor(temperature_std).float()
            self.parameters_mean  = torch.tensor(parameters_mean).float()
            self.parameters_std   = torch.tensor(parameters_std).float()
        self.length = 990

        if self.with_cluster:
            assert self.n_cluster in [0, 10, 20, 40, 30], f'Please, check if clustering has been computed offline ' \
                                                          f'for {self.n_cluster} nodes per cluster'

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        mesh_pos, faces, node_type, t, velocity, pressure, temperature, parameters = get_data(self.dataloc[item], self.window_length, self.mode, self.max_timestep)
        faces       = torch.from_numpy(faces).long()
        mesh_pos    = torch.from_numpy(mesh_pos).float()
        velocity    = torch.from_numpy(velocity).float()
        pressure    = torch.from_numpy(pressure).float()
        temperature = torch.from_numpy(temperature).float()
        edges       = faces_to_edges(faces)  # Convert triangles to edges (pairs of indices)
        node_type   = torch.from_numpy(node_type).long()
        parameters  = torch.from_numpy(parameters).float()

        if self.apply_onehot:
            node_type = one_hot(node_type, num_classes=9).squeeze(-2)

        if self.do_normalization:
            velocity, pressure, temperature, parameters = self.normalize(velocity, pressure, temperature, parameters)

        output = {'mesh_pos': mesh_pos,
                  'edges': edges,
                  'velocity': velocity,
                  'pressure': pressure,
                  'temperature': temperature,
                  'node_type': node_type,
                  'parameters': parameters}

        if self.with_cells:
            output['cells'] = faces

        if self.with_cluster:
            cluster_path = self.dataloc[item].replace("Eagle_dataset", "Eagle_cluster")
            assert os.path.exists(cluster_path), f'Pre-computed cluster are not found, check path: {cluster_path}.\n' \
                                                 f'or update the path in the Dataloader.'
            if self.n_cluster == 0:
                clusters = torch.arange(mesh_pos.shape[1] + 1).view(1, -1, 1).repeat(velocity.shape[0], 1, 1)
            else:
                clusters = np.load(os.path.join(cluster_path, f"constrained_kmeans_{self.n_cluster}.npy"),
                                   mmap_mode='r')[t:t + self.window_length].copy()
                clusters = torch.from_numpy(clusters).long()
            output['cluster'] = clusters
        return output


    def normalize(self, velocity=None, pressure=None, temperature=None, parameters=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            mean = self.pressure_mean.to(pressure.device)
            std  = self.pressure_std.to(pressure.device)
            pressure = pressure.reshape(-1, 2)
            pressure = (pressure - mean) / std
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = self.velocity_mean.to(velocity.device)
            std  = self.velocity_std.to(velocity.device)
            velocity = velocity.reshape(-1, 2)
            velocity = (velocity - mean) / std
            velocity = velocity.reshape(velocity_shape)
        if temperature is not None:
            temperature_shape = temperature.shape
            mean = self.temperature_mean.to(temperature.device)
            std  = self.temperature_std.to(velocity.device)
            temperature = (temperature - mean) / std
            temperature = temperature.reshape(temperature_shape)
        if parameters is not None:
            parameters_shape = parameters.shape
            mean = self.parameters_mean.to(parameters.device)
            std  = self.parameters_std.to(parameters.device)
            parameters = parameters.reshape(-1, 2)
            parameters = (parameters - mean) / std
            parameters = parameters.reshape(parameters_shape)

        return velocity, pressure, temperature, parameters

    def denormalize(self, velocity=None, pressure=None, temperature=None, parameters=None):
        if pressure is not None:
            pressure_shape = pressure.shape        
            mean = self.pressure_mean.to(pressure.device)
            std  = self.pressure_std.to(pressure.device)
            pressure = pressure.reshape(-1, 2)
            pressure = (pressure * std) + mean
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = self.velocity_mean.to(velocity.device)
            std  = self.velocity_std.to(velocity.device)
            velocity = velocity.reshape(-1, 2)
            velocity = velocity * std + mean
            velocity = velocity.reshape(velocity_shape)
        if temperature is not None:
            temperature_shape = temperature.shape
            mean = self.temperature_mean.to(temperature.device)
            std  = self.temperature_std.to(temperature.device)
            temperature = temperature * std + mean
            temperature = temperature.reshape(temperature_shape)
        if parameters is not None:
            parameters_shape = parameters.shape
            mean = self.parameters_mean.to(parameters.device)
            std  = self.parameters_std.to(parameters.device)
            parameters = parameters.reshape(-1, 2)
            parameters = parameters * std + mean
            parameters = parameters.reshape(parameters_shape)
        return velocity, pressure, temperature, parameters


def get_data(path, window_length, mode, max_timestep=50):
    # as we are working with different data sets, the data set length can be different
    data = np.load(os.path.join(path, 'sim.npz'), mmap_mode='r')
    max_length = data["pointcloud"].shape[0]

    # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
    #t = 0 if window_length == max_length else random.randint(0, max_length - window_length)
    t = 0 if window_length == max_length else random.randint(0, max_timestep)
    t = 0 if mode != "train" and window_length != max_length else t
    
    mesh_pos = data["pointcloud"][t:t + window_length].copy()

    cells = np.load("/" + os.path.join(path, f"triangles.npy"))
    cells = cells[t:t + window_length]

    Vx = data['VX'][t:t + window_length].copy()
    Vy = data['VY'][t:t + window_length].copy()

    Ps = data['PS'][t:t + window_length].copy()
    Pg = data['PG'][t:t + window_length].copy()

    temp = data['T'][t:t + window_length].copy()

    velocity  = np.stack([Vx, Vy], axis=-1)
    pressure  = np.stack([Ps, Pg], axis=-1)
    node_type = data['mask'][t:t + window_length].copy()

    Hc = np.ones((velocity.shape[0], velocity.shape[1], 1))*data['HC'][0]
    Tc = np.ones((velocity.shape[0], velocity.shape[1], 1))*data['TC'][0]
    parameters = np.stack((Hc, Tc), axis=-1).squeeze()

    return mesh_pos, cells, node_type, t, velocity, pressure, temp, parameters


def faces_to_edges(faces):
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)

    return unique_edges


def calculate_statistics(data_path):
    # calculate the statistics of the dataset
    stat_path = os.path.join(data_path, 'statistics.npz')
    if os.path.exists(stat_path):
        stat             = np.load(stat_path)
        pressure_mean    = stat['pressure_mean']
        pressure_std     = stat['pressure_std']
        velocity_mean    = stat['velocity_mean']
        velocity_std     = stat['velocity_std']
        temperature_mean = stat['temperature_mean']
        temperature_std  = stat['temperature_std']
        parameters_mean  = stat['parameters_mean']
        parameters_std   = stat['parameters_std']
    else:
        print('statistics not calculated yet, start calculating')
        # gather all data from the train, test, and validation sets
        modes = ["train"]
        dataloc = []
        for mode in modes:
            with open(os.path.join(data_path, f"Splits/{mode}.txt"), "r") as f:
                for line in f.readlines():
                    dataloc.append(os.path.join(data_path, line.strip()))
        eps = 1e-8
        n   = len(dataloc)

        pressure_mean    = np.zeros([n, 2], dtype=np.float64)
        pressure_std     = np.zeros([n, 2], dtype=np.float64)
        velocity_mean    = np.zeros([n, 2], dtype=np.float64)
        velocity_std     = np.zeros([n, 2], dtype=np.float64)
        temperature_mean = np.zeros([n, 1], dtype=np.float64)
        temperature_std  = np.zeros([n, 1], dtype=np.float64)
        parameters_mean  = np.zeros([n, 2], dtype=np.float64)
        parameters_std   = np.zeros([n, 2], dtype=np.float64)

        for i, f in enumerate(tqdm(dataloc)):
            data = np.load(os.path.join(f, 'sim.npz'))
            pressure_mean[i, 0] += np.mean(data['PS'], dtype=np.float64)
            pressure_std[i, 0]  += np.mean(data['PS']**2, dtype=np.float64)
            pressure_mean[i, 1] += np.mean(data['PG'], dtype=np.float64)
            pressure_std[i, 1]  += np.mean(data['PG']**2, dtype=np.float64)

            velocity_mean[i, 0] += np.mean(data['VX'], dtype=np.float64)
            velocity_std[i, 0]  += np.mean(data['VX']**2, dtype=np.float64)
            velocity_mean[i, 1] += np.mean(data['VY'], dtype=np.float64)
            velocity_std[i, 1]  += np.mean(data['VY']**2, dtype=np.float64)

            temperature_mean[i] += np.mean(data['T'], dtype=np.float32)
            temperature_std[i]  += np.mean(data['T']**2, dtype=np.float32)

            parameters_mean[i]  += np.array((data['HC'], data['TC'])).squeeze()
            parameters_std[i]   += np.array((data['HC']**2, data['TC']**2)).squeeze()
     
        pressure_mean    = np.mean(pressure_mean, axis=0)
        pressure_std     = np.mean(pressure_std,  axis=0)
        velocity_mean    = np.mean(velocity_mean, axis=0)
        velocity_std     = np.mean(velocity_std,  axis=0)
        temperature_mean = np.mean(temperature_mean, axis=0)
        temperature_std  = np.mean(temperature_std, axis=0)
        parameters_mean  = np.mean(parameters_mean, axis=0)
        parameters_std   = np.mean(parameters_std, axis=0)
        
        pressure_std     = np.maximum(np.sqrt(pressure_std - pressure_mean**2), eps)
        velocity_std     = np.maximum(np.sqrt(velocity_std - velocity_mean**2), eps)
        temperature_std  = np.maximum(np.sqrt(temperature_std - temperature_mean**2), eps)
        parameters_std   = np.maximum(np.sqrt(parameters_std - parameters_mean**2), eps)

        # save statistics for future use
        np.savez(stat_path, 
                 pressure_mean=pressure_mean, pressure_std=pressure_std, 
                 velocity_mean=velocity_mean, velocity_std=velocity_std,
                 temperature_mean=temperature_mean, temperature_std=temperature_std,
                 parameters_mean=parameters_mean, parameters_std=parameters_std)

    print('pressure_mean:    ' + str(pressure_mean))
    print('pressuer_std:     ' + str(pressure_std))
    print('velocity_mean:    ' + str(velocity_mean))
    print('velocity_std:     ' + str(velocity_std))
    print('temperature_mean: ' + str(temperature_mean))
    print('temperature_std:  ' + str(temperature_std))
    print('parameters_mean:  ' + str(parameters_mean))
    print('parameters_std:   ' + str(parameters_std))

    return pressure_mean, pressure_std, velocity_mean, velocity_std, temperature_mean, temperature_std, parameters_mean, parameters_std
