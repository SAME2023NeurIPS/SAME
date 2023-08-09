#! /bin/sh

# cd src || exit

explainer=$1
dataset_name=$2
max_ex_size=$3
sparsity=$4
model_name=$5

if [ "$explainer" = "subgraphX" ]; then
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, GNN: $model_name Sparsity: $sparsity"
    python SubgraphX.py models="$model_name" datasets="$dataset_name" \
    sparsity=$sparsity max_ex_size=$max_ex_size
elif [ "$explainer" = "same" ]; then
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, Sparsity: $sparsity"
    if  [ "$dataset_name" = "ba_shapes" ]; then
        python SAME_for_node.py models="$model_name" datasets="$dataset_name" \
        sparsity=$sparsity max_ex_size=$max_ex_size
    else
        python SAME.py models="$model_name"
    fi
elif [ "$explainer" = "gstarx" ] || [ "$explainer" = "orphicx" ] || [ "$explainer" = "graphsvx" ]; then
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, Sparsity: $sparsity"
    python run_"$explainer".py models=$model datasets=$datasets explainers.sparsity=$sparsity
else
    echo ">>>> Explainer: $explainer, Dataset: $dataset_name, Sparsity: $sparsity"
    if [ "$explainer" = "pgexplainer" ]; then
        python -m "$explainer"_edges models="$model_name" \
        datasets="$dataset_name" explainers="$explainer" sparsity=$sparsity
    else
        python -m "$explainer" models="$model_name" \
        datasets="$dataset_name" explainers="$explainer" sparsity=$sparsity
    fi
fi

