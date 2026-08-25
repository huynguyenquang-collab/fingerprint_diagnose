from pathlib import Path
import shutil


def require_free_bytes(path, required_bytes):
    free = shutil.disk_usage(Path(path)).free
    if free < required_bytes:
        raise RuntimeError(f"insufficient scratch space at {path}: need {required_bytes}, have {free}")
    return free


def model_memory_metadata(model):
    return {"parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
            "devices": sorted({str(parameter.device) for parameter in model.parameters()})}
