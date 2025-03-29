from dataclasses import dataclass
from typing import Dict, List
import torch.nn as nn
import numpy as np

"""
f(x) = ReLU(W1@x + b1) * w2 + b2
"""
class ComplexModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

@dataclass
class StepMetrics:
    step: int  # The current step number
    pattern_counts: Dict[str, float]  # e.g. {"2P2N_pos": 8, "2P2N_neg": 3}
    train_error: float
    test_error: float
    overlap_stats: Dict[str, float]  # The overlap statistics
    model: nn.Module

    @property
    def step_label(self) -> str:
        return f"{self.step}/5"

class ExperimentRun:
    def __init__(self, num_steps: int, num_features_per_clause: int, cset: list[list[tuple[int, bool]]]):
        self.steps: List[StepMetrics] = []
        self.num_features_per_clause = num_features_per_clause
        self.cset = cset
        
    def add_step(self, step: int, usage_stats: tuple, train_err: float, 
                 test_err: float, overlap_stats: dict, model: nn.Module) -> None:
        # Convert usage tuple into named dictionary
        pattern_counts = {}
        k = self.num_features_per_clause
        for i in range(k + 1):
            pattern_counts[f"{i}P{k-i}N_pos"] = usage_stats[(i, k-i, 1)]
            pattern_counts[f"{i}P{k-i}N_neg"] = usage_stats[(i, k-i, -1)]
            
        metrics = StepMetrics(
            step=step,
            pattern_counts=pattern_counts,
            train_error=train_err,
            test_error=test_err,
            overlap_stats=overlap_stats,
            model=model
        )
        self.steps.append(metrics)
    
    def get_pattern_history(self, pattern_name: str) -> np.ndarray:
        """Get the history of a specific pattern across all steps"""
        return np.array([step.pattern_counts[pattern_name] for step in self.steps])
    
    def get_error_history(self, error_type: str = "train") -> np.ndarray:
        """Get either train or test error history"""
        if error_type == "train":
            return np.array([step.train_error for step in self.steps])
        return np.array([step.test_error for step in self.steps])

    def get_step_values(self):
        """Get the step values"""
        return np.array([step.step for step in self.steps])