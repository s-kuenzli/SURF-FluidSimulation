# SURF: A Generalisation Benchmark for GNNs Predicting Fluid Dynamics
SURF is a benchmark designed to test the generalization of learned graph-based fluid simulators.

## Installation

conda environment:

conda create --name surf -c pytorch -c nvidia -c pyg python=3.10.11 numpy==1.23.5 pytorch=1.13.0 pytorch-cuda=11.7 pytorch-scatter=2.1.1  pytorch-sparse=0.6.17 tqdm=4.65.0

## Usage Example 1: Evaluate a Previously Trained Model (Error Calculation)
Evaluate the MGN model "MGNBase_seed0" for the test dataset "Base":

python train_mgn_diffloss.py --epoch=0 --name=MGNBase_seed0 --dataset_path=/scratch/Base --horizon_train=8 --n_processor=15

Evaluate the EAGLE model "GVBase_seed0" for the test data set "Base":

python train_graphvit.py --epoch=0 --name=GraphVitBase_seed0 --dataset_path=/scratch/Base --horizon_train=8 --n_cluster=20

The calculated erros are saved in *.csv files for the velocity, pressure and temperature for 1 up to 250 time predictions. Note, that the reported prediction error for example at time step 250 is already averaged over all predicted time steps. These errors can be used to calculate the reported generalization scores

## Usage Example 2: Evaluate Specific Design Points (Result Calculation e.g. for Plotting)

If we would like to save the predicted values we can use the following command:

python train_mgn_diffloss.py --epoch=0 --name=MGNBase_seed0 --dataset_path=/scratch/Base --horizon_train=8 --n_processor=15 --save=True

This evaluates all datapoints which are listed in /scratch/Base/Splits/save.txt and saves the predicted values in ../Predictions/MGN_Base_seed0/Base/ 

## Usage Example 3: Train a Model

Train the EAGLE model on the dataset "Topo" for 1000 epochs:

python train_graphvit.py --epoch=1000 --name=GraphVitTopo --dataset_path=/scratch/Topo --horizon_train=2 --n_cluster=20

The model checkpoints are then saved as ../Trained_Models/graphvit/GraphVitTopo.nn
