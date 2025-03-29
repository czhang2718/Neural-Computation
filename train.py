import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data import generate_dataset
from stats import (
    measure_aggregator_usage,
    measure_train_test_error,
    measure_homogenous_sets,
    measure_kp_overlap_stats,
    plot_6final_snapshots_2x3
)
from dataclasses import dataclass
from typing import Dict, List
from utils import ExperimentRun, ComplexModel, StepMetrics


def run_single_model(
    run_i: int,
    aggregator_steps: list[int],
    plot_steps: list[int],
    j_size: int,
    num_features_per_clause: int,
    cset: list[list[tuple[int, bool]]],
    train_loader: DataLoader,
    test_loader: DataLoader,
    run_name: str,
    base_dir: str=".",
    l2_reg_factor: float=0.0,
    l1_reg_factor: float=1e-4,
    seed_offset: int=0,
):
    """
    Trains and evaluates a model using a randomly chosen k-clause logical formula.

    This function runs a single experiment where:
      - A formula is randomly chosen from a global clause pool.
      - A model is trained to learn that formula on synthetic data.
      - Various intermediate metrics are collected and visualized.

    The function tracks how well the model learns and generalizes using
    specific clause-pattern statistics at partial training steps and plots 
    the trends over time.

    Parameters
    ----------
    run_i : int
        Index of the current run (used for seeding and naming).
    aggregator_steps : list[int]
        List of indices of the aggregator labels to be used.
    plot_steps : list[int]
        List of indices of the steps to be plotted.
    j_size : int
        Size of the hidden layer in the model.
    num_features_per_clause : int
        Number of features per clause in the formula.
    train_loader : DataLoader
        DataLoader for training data.
    test_loader : DataLoader
        DataLoader for test data.
    run_name : str
        Name of the run.
    base_dir : str
        Base directory for storing results. 
    l2_reg_factor : float
        L2 regularization factor.
    l1_reg_factor : float
        L1 regularization factor.
    seed_offset : int
        Seed offset for random number generation.
    
    Returns
    -------
    aggregator_run : np.ndarray
        Array storing intermediate and final aggregator statistics
        such as clause pattern counts, errors, and overlap metrics.
    """

    local_seed = 57 + seed_offset + 1000*run_i
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    B = len(list(train_loader))
    chunk_size = max(1, B//5)

    model = ComplexModel(32, j_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    prior_pos_sets = None
    prior_neg_sets = None

    # Initialize the experiment
    experiment = ExperimentRun(len(aggregator_steps), num_features_per_clause, cset)

    def measure_store(step, step_index):
        nonlocal prior_pos_sets, prior_neg_sets
        
        usage_ = measure_aggregator_usage(model, cset)
        tr_e, ts_e = measure_train_test_error(model, train_loader, test_loader)
        pos_sets, neg_sets = measure_homogenous_sets(model, cset)
        over_ = measure_kp_overlap_stats(pos_sets, neg_sets, prior_pos_sets, prior_neg_sets)
        
        # Store all metrics in one place
        experiment.add_step(step, usage_, tr_e, ts_e, over_, model)
        
        prior_pos_sets = pos_sets
        prior_neg_sets = neg_sets

        step_label = str(step) + "/5"

        print(f"{run_name}: epoch={step_label}, trainErr={tr_e:.4f}, testErr={ts_e:.4f}, "
              f"kP+={over_['total_kp_pos']}, kP-={over_['total_kp_neg']}")

    def do_train_chunk(n_):
        for _ in range(n_):
            bx, by = next(iter(train_loader))
            optimizer.zero_grad()
            out = model(bx).squeeze()
            loss = criterion(out, by)
            if l2_reg_factor > 0:
                sum_l2 = 0.0
                for param in model.parameters():
                    sum_l2 += torch.sum(param**2)
                loss += (l2_reg_factor * sum_l2)
            if l1_reg_factor > 0:
                sum_l1 = 0.0
                for param in model.parameters():
                    sum_l1 += torch.sum(torch.abs(param))
                loss += (l1_reg_factor * sum_l1)
            loss.backward()
            optimizer.step()

    measure_store(0, 0)
    # partial steps in aggregator labels
    for step_idx_, step in enumerate(aggregator_steps):
        step_idx = step_idx_ + 1
        num_chunks = step - aggregator_steps[step_idx_-1] if step_idx_ > 0 else step
        do_train_chunk(num_chunks * chunk_size)
        measure_store(step, step_idx)

    # aggregator lines
    xvals = experiment.get_step_values()
    step_labels = [str(step) + "/5" for step in aggregator_steps]
    plot_labels = [str(step) + "/5" for step in plot_steps]

    # Plot pattern counts
    fig, ax1 = plt.subplots(figsize=(10,6))
    colors = ["blue", "green"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    
    for i in range(num_features_per_clause + 1):
        pattern_pos = f"{i}P{num_features_per_clause-i}N_pos"
        pattern_neg = f"{i}P{num_features_per_clause-i}N_neg"
        ax1.plot(xvals, experiment.get_pattern_history(pattern_pos), "-", 
                 color=colors[0], marker=markers[i%len(markers)],
                 linewidth=2, label=f"{pattern_pos}")
        ax1.plot(xvals, experiment.get_pattern_history(pattern_neg), "-",
                 color=colors[1], marker=markers[i%len(markers)],
                 linewidth=2, label=f"{pattern_neg}")

    ax1.set_xlabel(f"Partial Steps {aggregator_steps}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Pattern Count/clause", fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1.3,0.5), fontsize=10)
    ax1.set_xticks(aggregator_steps)
    ax1.set_xticklabels(step_labels, rotation=45, fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(xvals, experiment.get_error_history("train"), "-o", 
             color="red", linewidth=2, markersize=6, label="Train BCE")
    ax2.set_ylabel("Train BCE Error", color="red", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc='upper right')

    plt.title(f"Run #{run_i+1} => Distinct Random {num_features_per_clause}-AND formula", fontsize=14, fontweight="bold")
    plt.tight_layout()

    outdir_pdf = os.path.join(base_dir,"pdfs")
    os.makedirs(outdir_pdf, exist_ok=True)
    outpdf = os.path.join(outdir_pdf, f"run_{run_i+1}_aggregator_lineplot.pdf")
    plt.savefig(outpdf, dpi=120)
    plt.show()
    plt.close(fig)

    plot_6final_snapshots_2x3(
        run_name=f"Random{num_features_per_clause} Run={run_i+1}",
        experiment=experiment,
        out_dir=outdir_pdf,
        final_keys=plot_labels
    )

    return experiment