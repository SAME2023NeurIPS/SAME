#! /bin/sh

# cd src || exit

explainer=$1
dataset_name=$2
sparsity=$3
model_name=$4

if [ "$explainer" = "subgraphx" ]; then
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, GNN: $model_name Sparsity: $sparsity"
    python baselines/SubgraphX.py models="$model_name" datasets="$dataset_name" \
    sparsity=$sparsity
elif [ "$explainer" = "same" ]; then
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, Sparsity: $sparsity"
    if  [ "$dataset_name" = "ba_shapes" ]; then
        python SAME_for_node.py models="$model_name" datasets="$dataset_name" \
        sparsity=$sparsity
    else
        python SAME.py models="$model_name" datasets="$dataset_name" \
        sparsity=$sparsity
    fi
elif [ "$explainer" = "gstarx" ] || [ "$explainer" = "orphicx" ] || [ "$explainer" = "graphsvx" ]; then
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, Sparsity: $sparsity"
    python baselines/run_"$explainer".py models=$model_name datasets=$dataset_name explainers.sparsity=$sparsity
else
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, Sparsity: $sparsity"
    if [ "$explainer" = "pgexplainer" ]; then
        python baselines/"$explainer"_edges.py models="$model_name" \
        datasets="$dataset_name" explainers="$explainer" sparsity=$sparsity
    else
        python baselines/"$explainer".py models="$model_name" \
        datasets="$dataset_name" explainers="$explainer" sparsity=$sparsity
    fi
fi

