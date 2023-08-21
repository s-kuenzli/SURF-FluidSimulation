import os
import torch
import torch.nn as nn
from Dataloader.eagle import EagleDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.MeshGraphNetDiffLoss import MeshGraphNetDiffLoss
import argparse
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--gamma', default=0.991, type=float, help="Multiplicative factor of learning rate decay")
parser.add_argument('--dataset_path', default='', type=str, help="Dataset location")
parser.add_argument('--alpha', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--alphaT', default=0.1, type=float, help="Weighting for the temperature term in the loss")
parser.add_argument('--batchsize', default=1, type=int, help="Batch size")
parser.add_argument('--horizon_val', default=25, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=6, type=int, help="Number of timestep to train on")
parser.add_argument('--n_processor', default=10, type=int, help="Number of chained GNN layers")
parser.add_argument('--noise_std', default=2e-2, type=float,
                    help="Standard deviation of the gaussian noise to add on the input during training")
parser.add_argument('--name', default='mgn', type=str, help="Name for model")
parser.add_argument('--output_name', default='', type=str, help="Name for saving RMSE error to")
parser.add_argument('--save', default=False, type=bool, help="wheter to save predictions for datapoints defined in save.txt")
parser.add_argument('--max_timestep', default=50, type=int, help="maximal timestep to use for training (avoid feeding constant results to training)")
parser.add_argument('--seed', default=0, type=int, help="seed value")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()
BATCHSIZE = 1

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2

if args.output_name == '':
    args.output_name = args.name

def evaluate(save=False):
    print(args)
    length = 251
    mode = "save" if save else "test"
    dataset = EagleDataset(args.dataset_path, mode=mode, window_length=length,
                           with_cluster=False, normalize=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = MeshGraphNetDiffLoss(apply_noise=True, state_size=5, parameter_size=2, N=args.n_processor).to(device)

    model.load_state_dict(torch.load(f"../Trained_Models/meshgraphnet/{args.name}.nn", map_location=device))

    if save:
        # create list of datalocations for saving the results of the model
        path_dst = []
        path_src = []
        datasetname = os.path.split(os.path.dirname(args.dataset_path))[-1]
        with open(args.dataset_path + f"/Splits/{mode}.txt", "r") as f:
            for line in f.readlines():
                folder = line.strip()
                path_src.append(os.path.join(args.dataset_path, folder))
                path_dst.append(os.path.join(f"../Predictions/meshgraphnet/{args.name}/{datasetname}", folder))


    with torch.no_grad():
        model.eval()
        model.apply_noise = False

        error_velocity    = torch.zeros(length - 1).to(device)
        error_pressure    = torch.zeros(length - 1).to(device)
        error_temperature = torch.zeros(length - 1).to(device)

        os.makedirs(f"../Results/meshgraphnet", exist_ok=True)
        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            mesh_pos    = x["mesh_pos"].to(device)
            edges       = x['edges'].to(device).long()
            velocity    = x["velocity"].to(device)
            pressure    = x["pressure"].to(device)
            temperature = x["temperature"].to(device).unsqueeze(-1)
            node_type   = x["node_type"].to(device)
            parameters  = x["parameters"].to(device)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state = torch.cat([velocity, pressure, temperature], dim=-1)
            state_hat, _, _ = model(mesh_pos, edges, state, node_type, parameters)
            
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
            rmse_velocity    = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1,
                                                                                 device=device)
            rmse_pressure    = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1,
                                                                                 device=device)
            rmse_temperature = torch.cumsum(rmse_temperature, dim=0) / torch.arange(1, rmse_temperature.shape[0] + 1,
                                                                                 device=device)            

            error_velocity    = error_velocity + rmse_velocity
            error_pressure    = error_pressure + rmse_pressure
            error_temperature = error_temperature + rmse_temperature

            if save:
                vx = state_hat[0, :, :, 0].cpu().numpy()
                vy = state_hat[0, :, :, 1].cpu().numpy()
                ps = state_hat[0, :, :, 2].cpu().numpy()
                pg = state_hat[0, :, :, 3].cpu().numpy()
                t  = state_hat[0, :, :, 4].cpu().numpy()
                mesh_pos_c = mesh_pos[0][:, :, :].cpu().numpy()

                # save predicted data
                os.makedirs(path_dst[i], exist_ok=True)
                shutil.copyfile(os.path.join(path_src[i], 'triangles.npy'), os.path.join(path_dst[i], 'triangles.npy'))
                np.savez(os.path.join(path_dst[i], 'sim.npz'),
                        pointcloud=mesh_pos_c, VX=vx, VY=vy, PS=ps, PG=pg, T=t)

        error_velocity    = error_velocity / len(dataloader)
        error_pressure    = error_pressure / len(dataloader)
        error_temperature = error_temperature / len(dataloader)
        
        if save:
            np.savetxt(f"../Predictions/meshgraphnet/{args.name}/{datasetname}/{args.output_name}_error_velocity.csv", error_velocity.cpu().numpy(), delimiter=",")
            np.savetxt(f"../Predictions/meshgraphnet/{args.name}/{datasetname}/{args.output_name}_error_pressure.csv", error_pressure.cpu().numpy(), delimiter=",")
            np.savetxt(f"../Predictions/meshgraphnet/{args.name}/{datasetname}/{args.output_name}_error_temperature.csv", error_temperature.cpu().numpy(), delimiter=",")
        else:
            np.savetxt(f"../Results/meshgraphnet/{args.output_name}_error_velocity.csv", error_velocity.cpu().numpy(), delimiter=",")
            np.savetxt(f"../Results/meshgraphnet/{args.output_name}_error_pressure.csv", error_pressure.cpu().numpy(), delimiter=",")
            np.savetxt(f"../Results/meshgraphnet/{args.output_name}_error_temperature.csv", error_temperature.cpu().numpy(), delimiter=",")


# not used in mgn
def collate(X):
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])

    for x in X:
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
    loss  = MSE(target[..., :2] * mask, output[..., :2] * mask)
    loss += args.alpha * MSE(target[..., 2:4] * mask, output[..., 2:4] * mask)
    loss += args.alphaT * MSE(target[..., 4] * mask[..., 0], output[..., 4] * mask[..., 0])
    
    return loss


def validate(model, dataloader, epoch=0, vizu=False):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        model.apply_noise = False
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            mesh_pos    = x["mesh_pos"].to(device)
            edges       = x['edges'].to(device).long()
            velocity    = x["velocity"].to(device)
            pressure    = x["pressure"].to(device)
            temperature = x["temperature"].to(device).unsqueeze(-1)
            node_type   = x["node_type"].to(device)
            parameters  = x["parameters"].to(device)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state = torch.cat([velocity, pressure, temperature], dim=-1)

            state_hat, output, target = model(mesh_pos, edges, state, node_type, parameters)

            loss = get_loss(output, target, mask)
            total_loss += loss
            cpt += mesh_pos.shape[0]

        model.apply_noise = True
    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():    
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    batchsize = BATCHSIZE

    name = args.name

    train_dataset = EagleDataset(args.dataset_path, mode="train", window_length=args.horizon_train, with_cluster=False, max_timestep=args.max_timestep)
    valid_dataset = EagleDataset(args.dataset_path, mode="valid", window_length=args.horizon_val, with_cluster=False, max_timestep=args.max_timestep)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, 
                                  num_workers=1, pin_memory=True) #, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, 
                                  num_workers=1, pin_memory=True) #, collate_fn=collate)

    model = MeshGraphNetDiffLoss(apply_noise=True, state_size=5, parameter_size=2, N=args.n_processor).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)

    memory = torch.inf
    loss   = 0.0

    for epoch in range(args.epoch):
        model.train()
        model.apply_noise = True

        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            mesh_pos    = x["mesh_pos"].to(device).float()
            edges       = x['edges'].to(device).long()
            velocity    = x["velocity"].to(device).float()
            pressure    = x["pressure"].to(device).float()
            temperature = x["temperature"].to(device).float().unsqueeze(-1)
            node_type   = x["node_type"].to(device).long()
            parameters  = x["parameters"].to(device).float()
            #mask = torch.ones_like(mesh_pos)[..., 0]
            mask = torch.logical_or(node_type[:, :, :, NODE_NORMAL] == 1, node_type[:, :, :, NODE_OUTPUT] == 1)

            state = torch.cat([velocity, pressure, temperature], dim=-1)

            if (torch.isnan(state).any()):
                print(f"nan detected in outer training loop {i}: input data")
            if (torch.isnan(parameters).any()):
                print(f"nan detected in outer training loop {i}: parameters")  

            mesh_pos   = mesh_pos[0].unsqueeze(0) 
            edges      = edges[0].unsqueeze(0)
            state      = state[0].unsqueeze(0)
            node_type  = node_type[0].unsqueeze(0)
            parameters = parameters[0].unsqueeze(0)

            state_hat, output, target = model(mesh_pos, edges, state, node_type, parameters)

            if (torch.isnan(output).any()):
                print(f"nan detected in outer training loop {i}: output")
                raise SystemExit(0)
            if (torch.isnan(target).any()):
                print(f"nan detected in outer training loop {i}: target")
                raise SystemExit(0)
            
            # not really nice, but MGN cant handle ghost nodes... => no merging of data sets
            if args.batchsize > 1:
                epoch_loss = get_loss(output, target, mask)
                loss += epoch_loss

                if i % args.batchsize == 0:
                    if epoch > 1:  # Wait to accumulate dataset stat...
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                    loss = 0.0

            else:
                loss = get_loss(output, target, mask)

                if epoch > 1:  # Wait to accumulate dataset stat...
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

        if scheduler.get_last_lr()[0] > 1e-6 and epoch > 1:
            scheduler.step()

        error = validate(model, valid_dataloader, epoch=epoch)


        if error < memory:
            memory = error
            os.makedirs(f"../trained_models/meshgraphnet/", exist_ok=True)
            torch.save(model.state_dict(), f"../trained_models/meshgraphnet/{args.name}.nn")
            print("Saved!")

    validate(model, valid_dataloader)


if __name__ == '__main__':
    if args.epoch == 0 or args.save:
        evaluate(args.save)
    else:
        main()
