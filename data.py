import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def create_mixed_clause_pool_random_disjoint(
    global_input_dim: int = 16,
    pool_size: int = 256,
    features_per_and: int = 2,
    seed: int=0,
):
    """
    Create a mixed clause pool with random disjoint clauses.

    Parameters:
    ----------
    global_input_dim : int
        n, the total number of input features.
    pool_size : int
        The number of clauses in the pool.
    features_per_and : int
        The number of features in each clause.
    seed : int, optional
        The seed for the random number generator.

    Returns:
    -------
    clauses : list
        A list of clauses, where each clause is a list of (index, negated) pairs.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    max_disjoint = global_input_dim // features_per_and
    disjoint_count = min(max_disjoint, pool_size)
    free_vars = list(range(global_input_dim))
    random.shuffle(free_vars)
    clauses = []
    used=0
    for _ in range(disjoint_count):
        if used+features_per_and> len(free_vars):
            break
        chosen= free_vars[used : used+features_per_and]
        used+= features_per_and
        c= [(v,False) for v in sorted(chosen)]
        clauses.append(c)

    remain= pool_size - len(clauses)
    for _ in range(remain):
        chosen= random.sample(range(global_input_dim), features_per_and)
        c= [(v,False) for v in sorted(chosen)]
        clauses.append(c)

    return clauses


def create_hidden_function_from_clauses(clauses, input_dim):
    """
    Parameters:
    ----------
    clauses : list of list of ints
        List of formula's clauses, eg. {(x_1, x_2, False), (x_3, x_4, False), ...}.
    input_dim : int
        number of variables x_1, ..., x_n

    Returns:
    -------
    function
        Evaluator of f on input x.
    """
    def hidden_func(x):
        res= torch.zeros(x.shape[0], dtype=torch.bool)
        for clause in clauses:
            lits=[]
            for (v,neg) in clause:
                want_1= 1 if not neg else 0
                check= (x[:,v]== want_1)
                lits.append(check)
            conj= torch.stack(lits, dim=1).all(dim=1)
            res|= conj
        return res.float()
    return hidden_func


def generate_range_dataset(
    f_func,
    input_dim,
    num_samples,
    clauses,
    min_true_vars,
    max_true_vars,
    reverse_negated=True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a dataset of inputs and outputs based on logical clauses.

    This uses a "complex negative sampling" approach:
        - For label=1: Pick a random clause, set those bits, and add random bits.
        - For label=0: Forcibly break a random clause and ensure no clause is satisfied.

    Parameters
    ----------
    f_func : callable
        A function that evaluates the formula and returns a label (0 or 1).
    input_dim : int
        The number of input features (i.e., number of variables x_1, ..., x_n).
    num_samples : int
        Number of (X, y) samples to generate.
    clauses : list
        List of clauses, where each clause is a list of (index, negated) pairs.
    min_true_vars : int
        Minimum number of variables set to 1 in each data point.
    max_true_vars : int
        Maximum number of variables set to 1 in each data point.
    reverse_negated : bool, optional
        If True, reset unused variables to 0 after assignment (default: True).

    Returns
    -------
    X : torch.Tensor
        Input tensor of shape [num_samples, input_dim].
    y : torch.Tensor
        Output tensor of shape [num_samples]
    """
    X = torch.zeros(num_samples, input_dim)
    y = torch.zeros(num_samples)

    for i in range(num_samples):
        label = random.choice([0, 1])
        n = random.randint(min_true_vars, max_true_vars)

        if label == 1:
            if not clauses:
                continue

            c_sel = random.choice(clauses)
            for feat, neg in c_sel:
                X[i, feat] = 1.0 if not neg else 0.0

            used = len(c_sel)
            needed = n - used
            if needed < 0:
                continue

            c_feats = {cc[0] for cc in c_sel}
            remain = list(set(range(input_dim)) - c_feats)

            if reverse_negated:
                for rr in remain:
                    X[i, rr] = 0.0

            if needed <= len(remain):
                add_ = random.sample(remain, needed)
                for aa in add_:
                    X[i, aa] = 1.0

        else:
            attempts = 0
            made_neg = False

            while attempts < 100 and not made_neg:
                c_ = random.choice(clauses)
                row_ = np.zeros(input_dim, dtype=np.float32)

                for feat, neg in c_:
                    row_[feat] = 1.0 if not neg else 0.0

                # Flip one literal to break the clause
                f_idx, fneg = random.choice(c_)
                row_[f_idx] = 0.0 if not fneg else 1.0

                used = len(c_)
                c_feats = {c2[0] for c2 in c_}
                cur_ones = int(row_.sum())
                needed = n - cur_ones

                if needed < 0:
                    attempts += 1
                    continue

                remain = list(set(range(input_dim)) - c_feats)

                if reverse_negated:
                    for rr in remain:
                        row_[rr] = 0.0

                if needed > len(remain):
                    attempts += 1
                    continue

                add_ = random.sample(remain, needed)
                for aa in add_:
                    row_[aa] = 1.0

                # Ensure no clause is satisfied
                satisfied = False
                for ccl in clauses:
                    want_all = True
                    for vid, nn in ccl:
                        val_needed = 1.0 if not nn else 0.0
                        if row_[vid] != val_needed:
                            want_all = False
                            break
                    if want_all:
                        satisfied = True
                        break

                if not satisfied:
                    X[i] = torch.from_numpy(row_)
                    made_neg = True

                attempts += 1

        y[i] = f_func(X[i].unsqueeze(0))

    return X, y

def generate_dataset(
    cset,
    train_size,
    test_size,
    min_true_vars,
    max_true_vars,
    batch_size: int=64,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Generate a dataset of inputs and outputs based on logical clauses.
    
    Parameters
    ----------
    cset : list
        List of clauses, where each clause is a list of (index, negated) pairs.
    train_size : int
        Number of training samples to generate.
    test_size : int
        Number of test samples to generate.
    min_true_vars : int
        Minimum number of variables set to 1 in each data point.
    max_true_vars : int
        Maximum number of variables set to 1 in each data point.
    batch_size : int
        Batch size for data loaders.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data.
    """
    f_func = create_hidden_function_from_clauses(cset, 32)
    X_train, y_train = generate_range_dataset(f_func, 32, train_size, cset, min_true_vars, max_true_vars)
    X_test, y_test = generate_range_dataset(f_func, 32, test_size, cset, min_true_vars, max_true_vars)

    perm = torch.randperm(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]
    perm_t = torch.randperm(X_test.shape[0])
    X_test = X_test[perm_t]
    y_test = y_test[perm_t]

    I_ = torch.eye(32)
    X_train_t = torch.mm(X_train, I_.t())
    X_test_t = torch.mm(X_test, I_.t())
    train_data = TensorDataset(X_train_t, y_train)
    test_data = TensorDataset(X_test_t, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader