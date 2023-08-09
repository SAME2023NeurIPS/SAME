import numpy as np
from torch_geometric.utils.loop import add_self_loops, add_remaining_self_loops
from torch_geometric.utils import to_undirected


def hard_edge_masks2coalition(data, hard_edge_masks, added_self_loop=True):
    if added_self_loop:  # hard_edge_masks was generated with self-loop
        edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)
    else:
        edge_index = data.edge_index
        
    edge_index = edge_index[:, hard_edge_masks.bool()]
    coalition = edge_index.view(-1).unique().tolist()
    return coalition


def fidelity_normalize_and_harmonic_mean(fidelity, inv_fidelity, sparsity):
    """
    The idea is similar to the F1 score, two measures are summarized to one through harmonic mean.

    Step1: normalize both scores with sparsity
        norm_fidelity = fidelity * sparsity
        norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    Step2: rescale both normalized scores from [-1, 1] to [0, 1]
        rescaled_fidelity = (1 + norm_fidelity) / 2
        rescaled_inv_fidelity = (1 - norm_inv_fidelity) / 2
    Step3: take the harmonic mean of two rescaled scores
        2 / (1/rescaled_fidelity + 1/rescaled_inv_fidelity)

    Simplifying these three steps gives the formula
    """
    norm_fidelity = fidelity * sparsity
    norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    harmonic_fidelity = (
        (1 + norm_fidelity)
        * (1 - norm_inv_fidelity)
        / (2 + norm_fidelity - norm_inv_fidelity)
    )
    return norm_fidelity, norm_inv_fidelity, harmonic_fidelity


def evaluate_related_preds(related_preds):
    """
    related_preds is a dictionary output from the DIG implementation of the baseline models,
    i.e. GNNExplainer, PGExplainer, and SubgraphX

    related_preds contains four keys as shown below
    """

    original_prob = related_preds["origin"]
    masked_prob = related_preds["masked"]
    maskout_prob = related_preds["maskout"]
    sparsity = related_preds["sparsity"]

    fidelity = original_prob - maskout_prob
    inv_fidelity = original_prob - masked_prob
    return fidelity, inv_fidelity, sparsity


def evaluate_related_preds_list(related_preds_list, logger=None):
    """
    Evaluate the DIG implementation of the baseline models,
    i.e. GNNExplainer, PGExplainer, and SubgraphX

    Args:
    related_preds_list (list of dicts): each entry is a related_preds dict output from a baseline model
    """
    f_list = []
    inv_f_list = []
    n_f_list = []
    n_inv_f_list = []
    sp_list = []
    h_f_list = []
    for related_preds in related_preds_list:
        f, inv_f, sp = evaluate_related_preds(related_preds)
        n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
        f_list += [f]
        inv_f_list += [inv_f]
        n_f_list += [n_f]
        n_inv_f_list += [n_inv_f]
        sp_list += [sp]
        h_f_list += [h_f]

    f_mean = np.mean(f_list).item()
    inv_f_mean = np.mean(inv_f_list).item()
    n_f_mean = np.mean(n_f_list).item()
    n_inv_f_mean = np.mean(n_inv_f_list).item()
    sp_mean = np.mean(sp_list).item()
    h_f_mean = np.mean(h_f_list).item()

    if logger is not None:
        logger.info(
            f"Fidelity Mean: {f_mean:.4f}\n"
            f"Inv-Fidelity Mean: {inv_f_mean:.4f}\n"
            f"Norm-Fidelity Mean: {n_f_mean:.4f}\n"
            f"Norm-Inv-Fidelity Mean: {n_inv_f_mean:.4f}\n"
            f"Sparsity Mean: {sp_mean:.4f}\n"
            f"Harmonic-Fidelity Mean: {h_f_mean:.4f}\n"
        )
    return sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean