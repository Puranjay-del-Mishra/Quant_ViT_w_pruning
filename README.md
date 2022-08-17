# Quat_ViT_w_pruning
Quantised ViT with pruning that performs image classification

Image classification is performed on the CIFAR-10 dataset and a file with the name 'data' is supposed to be made which contains the cifar-10 dataset before training.
 This project uses a conda environment and the following packages were used in it's production -
 1) Torch pruning 0.2.7
 2) Pytorch 1.11.0
 
 Pruning alone gives 81.2% test accuracy and combination of pruning and the 8 bit quantisation gives 77.2 % accuracy with the right hyperparameters and under 40 epochs. Change in pruning threshold might lead to the giving an error and the dimension fix for pruning then would be changed with the directions mentioned as comments in the code.
