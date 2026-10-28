import os
import yaml

import numpy as np
import torch
from typing import Dict, List, Literal, Optional, Union, Tuple, Any


# def to_torch(nested_data, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Any:
#     if isinstance(nested_data, dict):
#         return {k: to_torch(v, dtype=dtype, device=device) for k, v in nested_data.items()}
#     elif isinstance(nested_data, list):
#         return [to_torch(v, dtype=dtype, device=device) for v in nested_data]
#     elif isinstance(nested_data, np.ndarray):
#         tensor = torch.from_numpy(nested_data)
#         if dtype is not None:
#             tensor = tensor.to(dtype)
#         if device is not None:
#             tensor = tensor.to(device)
#         return tensor
#     else:
#         return nested_data

def to_cuda(nested_data, non_blocking: bool = True) -> Any:
    if isinstance(nested_data, dict):
        return {k: to_cuda(v, non_blocking=non_blocking) for k, v in nested_data.items()}
    elif isinstance(nested_data, list):
        return [to_cuda(v, non_blocking=non_blocking) for v in nested_data]
    elif isinstance(nested_data, torch.Tensor):
        return nested_data.cuda(non_blocking=non_blocking)
    else:
        return nested_data

def dict_unpack(d: Dict[str, Any], *keys: str) -> List[Any]:
    return [d[key] for key in keys]

def save_checkpoint(path, info, model, optimizer):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
    with open(os.path.join(path, "info.yaml"), "w") as f:
        yaml.dump(info, f)