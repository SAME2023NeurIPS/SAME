#! /bin/sh

# cd src || exit

#explainer=subgraphX
#dataset_name=bbbp
max_ex_size=5
#sparsity=0.85
#model_name=gat

# explainer=gnn_explainer
# bash run_one_model.sh $explainer $dataset_name 5 $sparsity $model_name

# explainer=gnn_lrp
# bash run_one_model.sh $explainer $dataset_name 5 0.8 $model_name

# explainer=gnn_lrp
# bash run_one_model.sh $explainer $dataset_name 5 0.8 $model_name

# explainer=grad_cam
# bash run_one_model.sh $explainer $dataset_name 5 $sparsity $model_name

for explainer in gnn_explainer pgexplainer gnn_lrp grad_cam deep_lift subgraphX gstarx orphicx graphsvx same
do
   for dataset_name in ba_2motifs bbbp graph_sst2 graph_sst5 ba_shapes mutag twitter bace
   do
       for model_name in gcn gin gat
       do
           if [ "$explainer" = "subgraphX" ] || [ "$explainer" = "same" ]; then
               for sparsity in 0.5 0.6 0.7 0.8
               do
                   bash run_one_model.sh $explainer $dataset_name $max_ex_size \
                   $sparsity $model_name
               done
           else
               for sparsity in 0.8 0.7 0.6 0.5
               do
                   bash run_one_model.sh $explainer $dataset_name $max_ex_size \
                   $sparsity $model_name
               done
           fi
       done
   done
done
