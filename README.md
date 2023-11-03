<h1 align="center">:fire:SAME:fire:</h1>
<p align="center">
<a href="https://same2023neurips.github.io/"><img src="https://img.shields.io/website-up-down-green-red/http/shields.io.svg"></a>
</p>

This repo contains PyTorch implementation of the paper: `SAME: Uncovering GNN Black Box with Structure-aware Shapley-based Multipiece Explanations` in 2023 NeurIPS.


## Getting Started

### Requirements
The recommended Python version is 3.8 or 3.9. (SAME is tested under Python 3.8, Pytorch 1.12.1 and PyG 2.1.0)

```bash
conda create -n same python==3.8
```

Then, please follow the [Pytorch Guidelines](https://pytorch.org/get-started/previous-versions/) to install Pytorch and [INSTALL PYG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install PyG. After that, install other packages using: 

```bash
pip install -r requirements.txt
```

### Datasets

For the datasets used in our paper, please download them manually and move them into the folder `datasets/`. The file name is **case sensitive** and please make sure that all of the file names are in **lowercase**. For more information about the datasets, please refer to `dataset.py`. 

Here we provide the link to download them. (`SAME_datasets` contains both raw and processed data. If you want raw data only, please kindly click `SAME_raw_datasets`) </br>
[SAME_datasets](https://drive.google.com/file/d/1VXfb6hnKZ3nBsKMJPqvjI4Nz0ikk8skT/view?usp=sharing)  FILE_ID=`1VXfb6hnKZ3nBsKMJPqvjI4Nz0ikk8skT` </br>
[SAME_raw_datasets](https://drive.google.com/file/d/1KqSHf6Xz_PXKNA6KwXucye1H18hKIMJ6/view?usp=drive_link)  FILE_ID=`1KqSHf6Xz_PXKNA6KwXucye1H18hKIMJ6` </br>

\[Easy for Terminal\] You can download the dataset using the following bash. Remember to replace `FILE_ID` with that shown above. 

```bash
# install gdown to down file from google drive
pip install gdown
# download file using FILE_ID
gdown https://drive.google.com/uc?id=FILE_ID
```

### Models

Please use the following `bash` script to train the GNN models. Our code contains the implementation of `GCN`, `GAT` and `GIN`. The trained checkpoints of these models will be saved in `checkpoints/`.

```bash
python train_gnns.py models='gcn' datasets='mutag'
```

### Usage

- Train GNNs before explaining.
  - The `models` argument can be chosen from `GCN`, `GAT` or `GIN`. And you can create your own GNN model in `getNets.py`. 
  - For other datasets and GNNs, please create the corresponding files in `config`. 

```bash
python train_gnns.py datasets='mutag' models='gin'
```

- One-Click run: use the following script to run the experiments thoroughly. The script will automatically run the experiments in the sparsity $[0.5,0.6,0.7,0.8]$. 

```bash
bash main.sh
```

- Otherwise, you can run one explainer to explain a trained GNN model
  - You can run other baseline methods by setting `explainer` as `gnn_explainer`, `pgexplainer`, `gnn_lrp`, `grad_cam`, `subgraphX`, `gstarx`, `orphicx`, and `graphsvx`

```bash
explainer=same
dataset_name=bbbp
sparsity=0.8
model_name=gcn
bash run_one_model.sh $explainer $dataset_name $sparsity $model_name
```



## Acknowledgement

Some baseline methods of our code are adapted from the following repositories.

https://github.com/ShichangZh/GStarX

https://github.com/divelab/DIG/tree/main/dig/xgraph


## Contact Us

If you have any questions, please feel free contact us: [ziyuanye9801@gmail.com](ziyuanye9801@gmail.com) and [rihanhuang.work@gmail](rihanhuang.work@gmail) or open an issue.  

