import os

import torch
import torch.distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel


def env_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def env_rank():
    return int(os.environ.get("RANK", "0"))


def env_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed_run():
    return env_world_size() > 1


def can_use_distributed(config):
    return (
        config.MULTI_GPU
        and torch.cuda.is_available()
        and torch_dist.is_available()
        and is_distributed_run()
    )


def init_distributed(config):
    if not can_use_distributed(config):
        return torch.device(config.DEVICE)

    local_rank = env_local_rank()
    if local_rank >= len(config.GPU_IDS):
        raise ValueError(
            f"LOCAL_RANK={local_rank} exceeds configured GPU_IDS={config.GPU_IDS}"
        )

    device = torch.device(f"cuda:{config.GPU_IDS[local_rank]}")
    torch.cuda.set_device(device)
    torch_dist.init_process_group(backend=config.DIST_BACKEND)
    return device


def destroy_distributed():
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


def is_main_process():
    return not (torch_dist.is_available() and torch_dist.is_initialized()) or torch_dist.get_rank() == 0


def get_rank():
    if torch_dist.is_available() and torch_dist.is_initialized():
        return torch_dist.get_rank()
    return 0


def get_world_size():
    if torch_dist.is_available() and torch_dist.is_initialized():
        return torch_dist.get_world_size()
    return 1


def barrier():
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.barrier()


def wrap_model(model, device, config):
    if not can_use_distributed(config):
        return model

    return DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def state_dict(model):
    return unwrap_model(model).state_dict()


def reduce_sum(value, device):
    tensor = torch.tensor(float(value), device=device)
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.SUM)
    return tensor.item()


def clean_state_dict_keys(model_state_dict):
    if not any(key.startswith("module.") for key in model_state_dict):
        return model_state_dict
    return {
        key.removeprefix("module."): value
        for key, value in model_state_dict.items()
    }


def load_model_state(model, checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint)
    unwrap_model(model).load_state_dict(clean_state_dict_keys(state))
