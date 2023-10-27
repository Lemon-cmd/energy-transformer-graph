# Energy Transformer
Submission code for NEURIPS 2023

# Installation
```
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note, it is important to read the official [Jax](https://github.com/google/jax) installation guide to properly enable GPU and for further details.

## Test the install by starting python and running the following code to check to see if GPU is enabled for Jax:
```
import jax
print(jax.local_devices())
```

## Setting up data
Fortunately, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) has provided awesome datasets and dataloaders which will automatically download datasets when code is ran. Simply change the provided dataset name for TUDataset or GNNBenchmark.
```
model_name = data_name = 'CIFAR10'
train_data = GNNBenchmarkDataset(root = '../data/', name = data_name, split = 'train')
```

## See if it works
Simply, navigate to the ***nbs*** folder for the provided [Jupyter](https://jupyter.org) notebooks to run the experiments.
```
./run_nb_inplace nbs/eval_cifar10.ipynb
```
## Training from scratch
Since there are a number of provided pretrained models, please ensure that such files are removed or stored in a different folder such that they won't be reloaded.
```
./run_nb_inplace nbs/cifar10.ipynb
```

## Pretrained Models
Some pretrained models (e.g., CIFAR10, MNIST, and ZINC) are provided in the ***saved\_models*** folder.