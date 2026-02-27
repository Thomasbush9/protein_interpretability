import torch
import numpy as np  



def sort_layer_activations(layer_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    For one layer activation tensor (B, L, D), compute mean |act| per node and sort by most active.
    Returns:
        act_sorted: (D,) activity values in descending order
        top_indices: (D,) node indices that achieve that order
    """
    node_activity = layer_tensor.abs().squeeze().mean(dim=0).float().cpu().numpy()  # (D,)
    top_indices = np.argsort(-node_activity)
    act_sorted = node_activity[top_indices]
    return act_sorted, top_indices


