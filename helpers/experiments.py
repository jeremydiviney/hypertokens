import time
import glob

import psutil
from torch import nn
import torch

import wandb


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB for the current device"""
    if not torch.cuda.is_available():
        return 0.0

    return float(torch.cuda.max_memory_allocated()) / 1e9  # Convert bytes to GB


def get_memory_gb() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)


def count_parameters(model: nn.Module) -> tuple[int, dict]:
    """
    Count total trainable parameters and parameters per layer
    Returns: (total_params, params_by_layer)
    """
    params_by_layer = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    total_params = sum(params_by_layer.values())

    # Format large numbers with commas
    formatted_layers = {name: f"{count:,}" for name, count in params_by_layer.items()}

    print(f"\nTotal trainable parameters: {total_params:,}")
    # print("\nParameters per layer:")
    # for name, count in formatted_layers.items():
    #     print(f"{name}: {count}")

    return total_params, params_by_layer


def save_project_files_as_artifact(wandb_run):
    """Save all Python files in project as artifacts"""
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"source_code_{wandb_run.id}",
        type="code",
        description="Source code for this run",
    )

    # Include all Python files
    include_pattern = "**/*.py"
    # Exclude patterns for specific directories
    exclude_patterns = [
        ".venv/**/*.py",
        ".vscode/**/*.py",
        "wandb/**/*.py",
        "**/__init__.py",
    ]

    # Get all Python files then filter out excluded paths
    python_files = set(glob.glob(include_pattern, recursive=True)) - set(
        file for pattern in exclude_patterns for file in glob.glob(pattern, recursive=True)
    )

    # Add files to artifact
    for file_path in python_files:
        artifact.add_file(file_path)

    # Log the artifact
    wandb_run.log_artifact(artifact)


def run_experiment(projectName, train_model, exp_name, config: dict) -> None:

    # Initialize wandb
    wandb.login(key="2a4c6ae7fe4efb074b06e1bb9eca12afba05e310")

    wandb.init(
        project=projectName,
        config=config,
        name=exp_name,
    )

    # Track time and memory
    start_time = time.time()

    # Save source code at start of run
    save_project_files_as_artifact(wandb.run)

    # Train model and get parameters count
    train_model = train_model(wandb)

    total_params, params_by_layer = count_parameters(train_model)

    # Log parameters count
    wandb.log({"total_parameters": total_params, "run_id": wandb.run.id})

    gpu_memory_usage = get_gpu_memory_gb()

    # Log time and memory metrics
    end_time = time.time()
    duration = end_time - start_time

    wandb.log(
        {
            "duration_seconds": duration,
            "duration_per_epoch_seconds": duration / config["epochs"],
            "gpu_memory_usage_gb": gpu_memory_usage,
        }
    )

    wandb.finish()
