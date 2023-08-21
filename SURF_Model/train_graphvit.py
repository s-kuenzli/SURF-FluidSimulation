import torch
import torch.nn as nn
from Dataloader.eagle import EagleDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.GraphViT import GraphViT
import argparse
from tqdm import tqdm
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--dataset_path', default="/Volumes/Elements/Fluent/Eagle_dataset", type=str,
                    help="Dataset path, caution, the cluster location is induced from this path, make sure this is Ok")
parser.add_argument('--horizon_val', default=25, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=6, type=int, help="Number of timestep to train on")
parser.add_argument('--n_cluster', default=20, type=int, help="Number of nodes per cluster. 0 means no clustering")
parser.add_argument('--w_size', default=512, type=int, help="Dimension of the latent representation of a cluster")
parser.add_argument('--alpha', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--alphaT', default=0.1, type=float, help="Weighting for the temperature term in the loss")
parser.add_argument('--alphaV', default=1.0, type=float, help="Weighting for the velocity term in the loss")
parser.add_argument('--batchsize', default=1, type=int, help="Batch size")
parser.add_argument('--save', default=False, type=bool, help="wheter to save predictions for datapoints defined in save.txt")
parser.add_argument('--name', default='', type=str, help="Name for saving/loading weights")
parser.add_argument('--output_name', default='', type=str, help="Name for saving RMSE error to")
parser.add_argument('--max_timestep', default=50, type=int, help="maximal timestep to use for training (avoid feeding constant results to training)")
parser.add_argument('--seed', default=0, type=int, help="seed value")
args = parser.parse_args()

BATCHSIZE = args.batchsize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()

if args.output_name == '':
    args.output_name = args.name


def evaluate(save=False):
    print(args)
    length = 251
    mode = "save" if save else "test"
    dataset = EagleDataset(args.dataset_path, mode=mode, window_length=length,
                           with_cluster=True, n_cluster=args.n_cluster, normalize=True, with_cells=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                            collate_fn=collate)
    model = GraphViT(state_size=5, w_size=args.w_size, parameter_size=2).to(device)

    model.load_state_dict(
        torch.load(f"../Trained_Models/graphvit/{args.name}.nn", map_location=device))
    
    if save:
        # create list of datalocations for saving the results of the model
        path_dst = []
        path_src = []
        datasetname = os.path.split(os.path.dirname(args.dataset_path))[-1]
        with open(args.dataset_path + f"/Splits/{mode}.txt", "r") as f:
            for line in f.readlines():
                folder = line.strip()
                path_src.append(os.path.join(args.dataset_path, folder))
                path_dst.append(os.path.join(f"../Predictions/graphvit/{args.name}/{datasetname}", folder))

    with torch.no_grad():
        model.eval()

        error_velocity    = torch.zeros(length - 1).to(device)
        error_pressure    = torch.zeros(length - 1).to(device)
        error_temperature = torch.zeros(length - 1).to(device)

        os.makedirs(f"../Results/graphvit", exist_ok=True)
        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            mesh_pos    = x["mesh_pos"].to(device)
            edges       = x['edges'].to(device).long()
            velocity    = x["velocity"].to(device)
            pressure    = x["pressure"].to(device)
            temperature = x["temperature"].to(device)
            node_type   = x["node_type"].to(device)
            parameters  = x["parameters"].to(device)
            mask        = x["mask"].to(device)
            clusters    = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            # velocity[batch, timestep, node, vx/vy]
            state = torch.cat([velocity, pressure, temperature], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask, parameters,
                                              apply_noise=False)

            state_hat[..., :2], state_hat[..., 2:4], state_hat[..., 4], _ = dataset.denormalize(state_hat[..., :2], state_hat[..., 2:4], state_hat[..., 4])

            velocity, pressure, temperature, _ = dataset.denormalize(velocity, pressure, temperature)

            # discard the init state values 't=0', error is only calculated on predicted values
            velocity    = velocity[:, 1:]
            pressure    = pressure[:, 1:]
            temperature = temperature[:, 1:]
            velocity_hat    = state_hat[:, 1:, :, :2]
            pressure_hat    = state_hat[:, 1:, :, 2:4]
            temperature_hat = state_hat[:, 1:, :, 4]
            mask = mask[:, 1:].unsqueeze(-1)

            rmse_velocity    = torch.sqrt((velocity[0] * mask[0] - velocity_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)
            rmse_pressure    = torch.sqrt((pressure[0] * mask[0] - pressure_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)
            rmse_temperature = torch.sqrt((temperature[0] * mask[0] - temperature_hat[0].unsqueeze(-1) * mask[0]).pow(2).mean(dim=-1)).mean(1)
            rmse_velocity    = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1, device=device)
            rmse_pressure    = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1, device=device)
            rmse_temperature = torch.cumsum(rmse_temperature, dim=0) / torch.arange(1, rmse_temperature.shape[0] + 1, device=device)            

            error_velocity    = error_velocity + rmse_velocity
            error_pressure    = error_pressure + rmse_pressure
            error_temperature = error_temperature + rmse_temperature

            # save predicted data
            if save:
                # get rid of the phantom node defined in the collate
                vx = state_hat[0, :, :-1, 0].cpu().numpy()
                vy = state_hat[0, :, :-1, 1].cpu().numpy()
                ps = state_hat[0, :, :-1, 2].cpu().numpy()
                pg = state_hat[0, :, :-1, 3].cpu().numpy()
                t  = state_hat[0, :, :-1, 4].cpu().numpy()

                mesh_pos_c = mesh_pos[0][:, :-1, :].cpu().numpy()
                os.makedirs(path_dst[i], exist_ok=True)
                shutil.copyfile(os.path.join(path_src[i], 'triangles.npy'), os.path.join(path_dst[i], 'triangles.npy'))
                np.savez(os.path.join(path_dst[i], 'sim.npz'),
                        pointcloud=mesh_pos_c, VX=vx, VY=vy, PS=ps, PG=pg, T=t)                          

    error_velocity    = error_velocity / len(dataloader)
    error_pressure    = error_pressure / len(dataloader)
    error_temperature = error_temperature / len(dataloader)
    if save:
        np.savetxt(f"../Predictions/graphvit/{args.name}/{datasetname}/{args.output_name}_error_velocity.csv", error_velocity.cpu().numpy(), delimiter=",")
        np.savetxt(f"../Predictions/graphvit/{args.name}/{datasetname}/{args.output_name}_error_pressure.csv", error_pressure.cpu().numpy(), delimiter=",")
        np.savetxt(f"../Predictions/graphvit/{args.name}/{datasetname}/{args.output_name}_error_temperature.csv", error_temperature.cpu().numpy(), delimiter=",")
    else:
        np.savetxt(f"../Results/graphvit/{args.output_name}_error_velocity.csv", error_velocity.cpu().numpy(), delimiter=",")
        np.savetxt(f"../Results/graphvit/{args.output_name}_error_pressure.csv", error_pressure.cpu().numpy(), delimiter=",")
        np.savetxt(f"../Results/graphvit/{args.output_name}_error_temperature.csv", error_temperature.cpu().numpy(), delimiter=",")


def collate(X):
    """ Convoluted function to stack simulations together in a batch. Basically, we add ghost nodes
    and ghost edges so that each sim has the same dim. This is useless when batchsize=1 though..."""
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])
    C_max = max([x["cluster"].shape[-2] for x in X])

    for batch, x in enumerate(X):
        # This step add fantom nodes to reach N_max + 1 nodes
        for key in ['mesh_pos', 'velocity', 'pressure', 'parameters']:
            tensor = x[key]
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)

        tensor = x["temperature"]
        T, N = tensor.shape
        x["temperature"] = torch.cat([tensor, torch.zeros(T, N_max - N + 1)], dim=1)
        x["temperature"] = torch.unsqueeze(x["temperature"], -1)

        tensor = x["node_type"]
        T, N, S = tensor.shape
        x["node_type"] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

        x["cluster_mask"] = torch.ones_like(x["cluster"])
        x["cluster_mask"][x["cluster"] == -1] = 0
        x["cluster"][x["cluster"] == -1] = N_max

        if x["cluster"].shape[1] < C_max:
            c = x["cluster"].shape[1]
            x["cluster"] = torch.cat(
                [x["cluster"], N_max * torch.ones(x["cluster"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)
            x["cluster_mask"] = torch.cat(
                [x["cluster_mask"], torch.zeros(x["cluster_mask"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)

        edges = x['edges']
        T, E, S = edges.shape
        x['edges'] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)

        x['mask'] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

    output = {key: None for key in X[0].keys()}
    for key in output.keys():
        if key != "example":
            output[key] = torch.stack([x[key] for x in X], dim=0)
        else:
            output[key] = [x[key] for x in X]

    return output


def get_loss(output, target, mask):
    mask = mask[:, 1:].unsqueeze(-1)
    loss  = args.alphaV * MSE(target[..., :2] * mask, output[..., :2] * mask)
    loss += args.alpha * MSE(target[..., 2:4] * mask, output[..., 2:4] * mask)
    loss += args.alphaT * MSE(target[..., 4] * mask[..., 0], output[..., 4] * mask[..., 0])
    
    return loss


def validate(model, dataloader, epoch=0, vizu=False):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            mesh_pos    = x["mesh_pos"].to(device)
            edges       = x['edges'].to(device).long()
            velocity    = x["velocity"].to(device)
            pressure    = x["pressure"].to(device)
            temperature = x["temperature"].to(device)
            node_type   = x["node_type"].to(device)
            parameters  = x["parameters"].to(device)
            # the mask generated in collate is just ones... does not exclude boundary nodes (wall, inlet, outlet)
            mask        = x["mask"].to(device)
            clusters    = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure, temperature], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask, parameters,
                                              apply_noise=False)

            loss = get_loss(output, target, mask)
            total_loss += loss.item()
            cpt += 1 

    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    name = args.name

    train_dataset = EagleDataset(args.dataset_path, mode="train", window_length=args.horizon_train, with_cluster=True,
                                 n_cluster=args.n_cluster, normalize=True, max_timestep=args.max_timestep)
    valid_dataset = EagleDataset(args.dataset_path, mode="valid", window_length=args.horizon_val, with_cluster=True,
                                 n_cluster=args.n_cluster, normalize=True, max_timestep=args.max_timestep)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=1,
                                  pin_memory=False, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=1,
                                  pin_memory=True, collate_fn=collate)

    model = GraphViT(state_size=5, w_size=args.w_size).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    memory = torch.inf
    for epoch in range(args.epoch):
        model.train()

        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            mesh_pos    = x["mesh_pos"].to(device)
            edges       = x['edges'].to(device).long()
            velocity    = x["velocity"].to(device)
            pressure    = x["pressure"].to(device)
            temperature = x["temperature"].to(device)
            node_type   = x["node_type"].to(device)
            parameters  = x["parameters"].to(device)
            mask        = x["mask"].to(device)
            clusters    = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure, temperature], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask, parameters,
                                              apply_noise=True)

            loss = get_loss(output, target, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()

        error = validate(model, valid_dataloader, epoch=epoch)

        if error < memory:
            memory = error
            os.makedirs(f"../trained_models/graphvit/", exist_ok=True)
            torch.save(model.state_dict(), f"../trained_models/graphvit/{name}.nn")
            print("Saved!")

    validate(model, valid_dataloader)


if __name__ == '__main__':
    if args.epoch == 0 or args.save:
        evaluate(args.save)
    else:
        main()
