from collections import deque
from typing import Dict

N_METRICS = 120  # Number of metric samples to keep


def update_metric(metrics_dict: Dict, key: str, value: float, max_len: int = N_METRICS):
    """
    Helper to initialize and update a metric in the shared dictionary
    using a deque for efficiency.
    """
    if key not in metrics_dict:
        metrics_dict[key] = deque(maxlen=max_len)
    metrics_dict[key].append(value)
