import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.gridspec as gridspec
from utils import ExperimentRun


##############################
# (D) aggregator usage + train/test error
##############################
def measure_aggregator_usage(model, clauses):
    """
    Returns (kP0N_pos, kP0N_neg, (k-1)P1N_pos, (k-1)P1N_neg, ..., 0PkN_pos, 0PkN_neg)
    where k is the number of features in each clause.
    For each clause, counts neurons that have pos_count=k,k-1,...,0 among the k features
    with w2>0 or w2<0.
    """
    fc1_w = model.fc1.weight.detach().cpu().numpy()
    fc2_w = model.fc2.weight.detach().cpu().numpy()[0]
    
    # Initialize counters for each possible positive count (0 to k)
    counters = {}
    for pos_count in range(len(clauses[0]) + 1):
        counters[(pos_count, len(clauses[0])-pos_count, 1)] = 0
        counters[(pos_count, len(clauses[0])-pos_count, -1)] = 0
    
    for clause in clauses:
        sel_idx = [v for (v,neg) in clause]
        k = len(sel_idx)
        for neuron in range(fc1_w.shape[0]):
            w_sub = fc1_w[neuron, sel_idx]
            pos_count = np.sum(w_sub > 0)
            w2_val = fc2_w[neuron]
            if w2_val > 0:
                counters[(pos_count, k-pos_count, 1)] += 1
            else:
                counters[(pos_count, k-pos_count, -1)] += 1
    
    return counters

def measure_train_test_error(model, loader_train, loader_test):
    criterion= nn.BCEWithLogitsLoss()
    model.eval()
    tot=0.0
    cc=0
    with torch.no_grad():
        for bx,by in loader_train:
            out= model(bx).squeeze()
            loss= criterion(out, by)
            tot+= loss.item()
            cc+=1
    train_err= tot/cc if cc>0 else 0.0

    wrong=0
    total=0
    with torch.no_grad():
        for bx,by in loader_test:
            out= model(bx).squeeze()
            preds= (out>0).float()
            diffs= (preds!= by).float()
            wrong+= diffs.sum().item()
            total+= bx.shape[0]
    test_err= (wrong/ total) if total>0 else 0.0
    if test_err<=0:
        test_err= 1e-9
    return (train_err, test_err)


##############################
# (D2) measure sets of kP pos or kP neg for each clause, for overlap stats
##############################


def measure_homogenous_sets(
    model: nn.Module,
    clauses: list[list[tuple[int, bool]]],
    k: int|None=None,
) -> tuple[list[set[int]], list[set[int]]]:
    """
    Return the sets of neurons that are homogenous for each clause, ie.

    pos_sets = [
        [i st clause 1 has w1[i, :]=++...+, w2[i]=+],
        [i st clause 2 has w1[i, :]=++...+, w2[i]=+],
        ...
    ]
    neg_sets = [
        [i st clause 1 has w1[i, :]=++...+, w2[i]=-],
        [i st clause 2 has w1[i, :]=++...+, w2[i]=-],
        ...
    ]
    """
    fc1_w = model.fc1.weight.detach().cpu().numpy()    # shape => [hidden_dim,32]
    fc2_w = model.fc2.weight.detach().cpu().numpy()[0] # shape => [hidden_dim,]
    hidden_dim = fc1_w.shape[0]

    pos_sets = []
    neg_sets = []
    for i, clause in enumerate(clauses):
        sel_idx = [v for (v,n) in clause]
        if k is None:
            k = len(sel_idx)
        assert k == len(sel_idx), f"Clause {i} has {len(sel_idx)} features, expected {k}"
        pos_neurons = set()
        neg_neurons = set()
        for neuron in range(hidden_dim):
            w_sub = fc1_w[neuron, sel_idx]
            pos_count = np.sum(w_sub > 0)
            w2_val = fc2_w[neuron]
            if pos_count == k:  # All positive weights
                if w2_val > 0:
                    pos_neurons.add(neuron)
                else:
                    neg_neurons.add(neuron)
        pos_sets.append(pos_neurons)
        neg_sets.append(neg_neurons)
    return pos_sets, neg_sets

def measure_kp_overlap_stats(
    pos_sets: list[set[int]],
    neg_sets: list[set[int]],
    prior_pos_sets: list[set[int]]|None=None,
    prior_neg_sets: list[set[int]]|None=None,
) -> dict[str, int]:
    """
    Calculate overlap statistics for k-and clauses where k is the number of features in each clause.
    """
    assert (prior_pos_sets is None) == (prior_neg_sets is None), "prior_pos_sets and prior_neg_sets must both be None or both be provided"
    ccount = len(pos_sets)
    
    # Calculate total kP neurons and overlaps for positive w2
    total_kp_pos = sum(len(s) for s in pos_sets)
    sum_of_clauseOverlaps_kp_pos = 0
    for i in range(ccount):
        for j in range(i+1, ccount):
            sum_of_clauseOverlaps_kp_pos += len(pos_sets[i].intersection(pos_sets[j]))

    if prior_pos_sets is not None:
        overlap_kp_pos_with_prior = 0
        for i in range(ccount):
            overlap_kp_pos_with_prior += len(pos_sets[i].intersection(prior_pos_sets[i]))
    else:
        overlap_kp_pos_with_prior = 0

    # Calculate total kp neurons and overlaps for negative w2
    total_kp_neg = sum(len(s) for s in neg_sets)
    sum_of_clauseOverlaps_kp_neg = 0
    for i in range(ccount):
        for j in range(i+1, ccount):
            sum_of_clauseOverlaps_kp_neg += len(neg_sets[i].intersection(neg_sets[j]))

    if prior_neg_sets is not None:
        overlap_kp_neg_with_prior = 0
        for i in range(ccount):
            overlap_kp_neg_with_prior += len(neg_sets[i].intersection(prior_neg_sets[i]))
    else:
        overlap_kp_neg_with_prior = 0

    return {
        "total_kp_pos": total_kp_pos,
        "sum_of_clauseOverlaps_kp_pos": sum_of_clauseOverlaps_kp_pos,
        "overlap_kp_pos_with_prior": overlap_kp_pos_with_prior,
        "total_kp_neg": total_kp_neg,
        "sum_of_clauseOverlaps_kp_neg": sum_of_clauseOverlaps_kp_neg,
        "overlap_kp_neg_with_prior": overlap_kp_neg_with_prior
    }

    # TODO: generalize to return a dict of all (cnt_p, cnt_n)

##############################
# Reorder / plot snapshots
##############################
def reorder_cols_by_clause(
    fc1_w: np.ndarray,
    clauses: list[list[tuple[int, bool]]],
    max_clauses: int=8,
) -> tuple[np.ndarray, list[str]]:
    """
    Reorder the columns of fc1_w by the clause indices. Only for formulas with disjoint clauses.
    """
    assert 0 < len(clauses) <= max_clauses, f"Expected 0 < len(clauses) <= {max_clauses}, got {len(clauses)}"
    k = len(clauses[0])
    k_clauses= [c_ for c_ in clauses if len(c_)==k]
    # print(f"k_clauses: {k_clauses}")
    k_clauses= k_clauses[:max_clauses]
    used_cols=[]
    used_labels=[]
    for c_ in k_clauses:
        var_idx_list= [v for (v,neg) in c_]
        var_idx_list= sorted(var_idx_list)
        for v_ in var_idx_list:
            used_cols.append(v_)
            used_labels.append(str(v_))
    if len(used_cols)> fc1_w.shape[1]:
        used_cols= used_cols[: fc1_w.shape[1]]
        used_labels= used_labels[: fc1_w.shape[1]]
    W_reordered= fc1_w[:, used_cols]
    return W_reordered, used_labels

def reorder_rows_by_l2pos(W_l1, fc2_w):
    idx= np.arange(W_l1.shape[0])
    pos_idx= np.array([i for i in idx if fc2_w[i]>0])
    neg_idx= np.array([i for i in idx if fc2_w[i]<=0])
    if len(pos_idx)>0:
        sorted_pos= pos_idx[np.argsort(fc2_w[pos_idx])[::-1]]
    else:
        sorted_pos= np.array([], dtype=int)
    if len(neg_idx)>0:
        sorted_neg= neg_idx[np.argsort(fc2_w[neg_idx])[::-1]]
    else:
        sorted_neg= np.array([], dtype=int)
    new_order= np.concatenate((sorted_pos, sorted_neg), axis=0)
    W_rows= W_l1[new_order,:]
    w2_new= fc2_w[new_order]
    # print("W_rows", W_rows)
    return W_rows, w2_new, new_order

def plot_single_snapshot(
    fc1_w: np.ndarray,
    fc2_w: np.ndarray,
    step_label: str,
    clauses: list[list[tuple[int, bool]]],
    model_name: str,
    out_dir: str="plots/",
):
    """
    Plot a single snapshot of the model.
    """
    W_cols, xlabels= reorder_cols_by_clause(fc1_w, clauses, max_clauses=8)
    W_rows, w2_rows, _= reorder_rows_by_l2pos(W_cols, fc2_w)
    pos_count= sum(w2_rows>0)

    fig= plt.figure(figsize=(8,4))
    fig.suptitle(f"{model_name}, step={step_label}", fontsize=14, fontweight="bold")
    gs= gridspec.GridSpec(1,2, width_ratios=[3,0.5], wspace=0.05)

    ax_l1= fig.add_subplot(gs[0,0])
    cmax= abs(W_rows).max()
    im1= ax_l1.imshow(W_rows, cmap="bwr_r", aspect="auto", origin="upper", vmin=-cmax, vmax=cmax)
    ax_l1.set_yticks([])
    ax_l1.axhline(y= pos_count-0.5, color='k', linestyle='--')
    ax_l1.set_xticks(np.arange(len(xlabels)))
    ax_l1.set_xticklabels(xlabels, rotation=45, fontsize=8)
    ax_l1.set_title("W1", fontsize=10, fontweight="bold")

    ax_l2= fig.add_subplot(gs[0,1])
    w2_resh= w2_rows.reshape(-1,1)
    c2max= abs(w2_resh).max()
    im2= ax_l2.imshow(w2_resh, cmap="bwr_r", aspect="auto", origin="upper", vmin=-c2max, vmax=c2max)
    ax_l2.set_yticks([])
    ax_l2.set_xticks([])
    ax_l2.axhline(y= pos_count-0.5, color='k', linestyle='--')
    ax_l2.set_title("W2", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

    outfname= os.path.join(out_dir, f"snap_{model_name}_step_{step_label.replace('/','_')}.pdf")
    fig.savefig(outfname, dpi=120)
    plt.close(fig)
    print(f"Saved => {outfname}\n")


##############################
# We'll store the 6 special snapshots => eg. [0/5,3/5,6/5,10/5,15/5,100/5]
##############################

def plot_6final_snapshots_2x3(
    experiment: ExperimentRun,
    final_keys: list[str],
    run_name: str,
    out_dir: str="plots/",
):
    """
    Plot the 6 final snapshots.
    """
    fig= plt.figure(figsize=(24,10))  # bigger subplots
    fig.suptitle(f"{run_name} => final 2×3 snapshots", fontsize=18, fontweight="bold")
    outer_gs= gridspec.GridSpec(2,3, wspace=0.5, hspace=0.2)

    idx=0
    for r_ in range(2):
        for c_ in range(3):
            if idx>= len(final_keys):
                break
            step_label= final_keys[idx]
            idx+=1
    
            fc1_w = experiment.steps[idx].model.fc1.weight.detach().cpu().numpy()
            fc2_w = experiment.steps[idx].model.fc2.weight.detach().cpu().numpy()[0]
            b1_w = experiment.steps[idx].model.fc1.bias.detach().cpu().numpy()
            clauses = experiment.cset

            W_cols, xlabels = reorder_cols_by_clause(fc1_w, clauses)
            W_rows, w2_rows, _ = reorder_rows_by_l2pos(W_cols, fc2_w)
            b1_rows = b1_w[np.argsort(-w2_rows)]  # Reorder b1_w using same ordering as rows
            pos_count= sum(w2_rows>0)

            in_gs= gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[3, 0.3, 0.5], subplot_spec=outer_gs[r_,c_], wspace=0.05)
            ax_l1= fig.add_subplot(in_gs[0,0])
            cmax= abs(W_rows).max()
            im1= ax_l1.imshow(W_rows, cmap="bwr_r", aspect="auto", origin="upper", vmin=-cmax, vmax=cmax)
            # Add text annotations for Layer1
            for i in range(W_rows.shape[0]):
                for j in range(W_rows.shape[1]):
                    text = ax_l1.text(j, i, f'{W_rows[i,j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=6)
            ax_l1.set_title(f"Step={step_label}\nW1", fontsize=12, fontweight="bold")
            ax_l1.set_yticks([])
            ax_l1.axhline(y= pos_count-0.5, color='k', linestyle='--')
            ax_l1.set_xticks(np.arange(len(xlabels)))
            ax_l1.set_xticklabels(xlabels, rotation=45, fontsize=9)

            # Add b1_w plot
            ax_b1 = fig.add_subplot(in_gs[0,1])
            b1_resh = b1_rows.reshape(-1,1)
            b1max = abs(b1_resh).max()
            im_b1 = ax_b1.imshow(b1_resh, cmap="bwr_r", aspect="auto", origin="upper", vmin=-b1max, vmax=b1max)
            # Add text annotations for Bias1
            for i in range(b1_resh.shape[0]):
                text = ax_b1.text(0, i, f'{b1_resh[i,0]:.1f}',
                               ha="center", va="center", color="black", fontsize=6)
            ax_b1.set_title("b1", fontsize=10, fontweight="bold")
            ax_b1.set_yticks([])
            ax_b1.set_xticks([])
            ax_b1.axhline(y= pos_count-0.5, color='k', linestyle='--')

            ax_l2= fig.add_subplot(in_gs[0,2])
            w2_resh= w2_rows.reshape(-1,1)
            c2max= abs(w2_resh).max()
            im2= ax_l2.imshow(w2_resh, cmap="bwr_r", aspect="auto", origin="upper", vmin=-c2max, vmax=c2max)
            # Add text annotations for Layer2
            for i in range(w2_resh.shape[0]):
                text = ax_l2.text(0, i, f'{w2_resh[i,0]:.1f}',
                               ha="center", va="center", color="black", fontsize=6)
            ax_l2.set_title("W2", fontsize=10, fontweight="bold")
            ax_l2.set_yticks([])
            ax_l2.set_xticks([])
            ax_l2.axhline(y= pos_count-0.5, color='k', linestyle='--')

    fig.tight_layout(rect=[0,0,1,0.93])
    outpdf= os.path.join(out_dir, f"{run_name}_final_6snap_2x3.pdf")
    plt.savefig(outpdf, dpi=120)
    plt.show()
    plt.close(fig)
    print(f"Saved => {outpdf}")