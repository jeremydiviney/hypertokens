import wandb
import time
import psutil
import torch.nn as nn
import torch
from typing import TypedDict
import glob
from datetime import datetime

# Define experiment configuration type
class ExperimentConfig(TypedDict):
    seq_len: int
    encode_last_n_length: int
    hypertoken_size: int
    epochs: int
    batch_size: int
    lr: float
    n_heads: int
    n_layers: int
    embed_dim: int

def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB for the current device"""
    if not torch.cuda.is_available():
        return 0.0

    return float(torch.cuda.max_memory_allocated())/1e9  # Convert bytes to GB


def get_memory_gb() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)



def count_parameters(model: nn.Module) -> tuple[int, dict]:
    """
    Count total trainable parameters and parameters per layer
    Returns: (total_params, params_by_layer)
    """
    params_by_layer = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    total_params = sum(params_by_layer.values())
    
    # Format large numbers with commas
    formatted_layers = {
        name: f"{count:,}" for name, count in params_by_layer.items()
    }
    
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
        description="Source code for this run"
    )
    
    # Include all Python files
    include_pattern = '**/*.py'
    # Exclude patterns for specific directories
    exclude_patterns = [
        '.venv/**/*.py',
        '.vscode/**/*.py',
        'wandb/**/*.py'
    ]
    
    # Get all Python files then filter out excluded paths
    python_files = set(glob.glob(include_pattern, recursive=True)) - set(
        file
        for pattern in exclude_patterns
        for file in glob.glob(pattern, recursive=True)
    )
    
    # Add files to artifact
    for file_path in python_files:
        artifact.add_file(file_path)
    
    # Add all Python files to artifact
    for file_path in python_files:
        artifact.add_file(file_path)
    
    # Log the artifact
    wandb_run.log_artifact(artifact)

def run_experiment(projectName,model,train_model,dataloader,val_dataloader,config: ExperimentConfig) -> None:

    # Initialize wandb
    wandb.login(key="2a4c6ae7fe4efb074b06e1bb9eca12afba05e310")

    wandb.init(
        project=projectName,
        config=config,
        name=f"{projectName}-sl:{config['seq_len']}-elnl:{config['encode_last_n_length']}-hts:{config['hypertoken_size']}-e:{config['epochs']}-bs:{config['batch_size']}-lr:{config['lr']}-hs:{config['head_size']}-nl:{config['n_layers']}-ed:{config['embed_dim']}-cf:{config['compress_factor']}",
        )
    
    wandb.log({
        "run_id": wandb.run.id
    })

    # Track time and memory
    start_time = time.time()
   
    try:

        # Save source code at start of run
        save_project_files_as_artifact(wandb.run)

        # Train model and get parameters count
        model = train_model(wandb,model,dataloader,val_dataloader,config)
        
        total_params, params_by_layer = count_parameters(model)

        # Log parameters count
        wandb.log({
            "total_parameters": total_params,
        })

      
    
        memory_usage = model.average_memory_usage

        gpu_memory_usage = get_gpu_memory_gb()
    except Exception as e:
        print(e)

    finally:
        # Log time and memory metrics
        end_time = time.time()
        duration = end_time - start_time
        
        wandb.log({
            "duration_seconds": duration,
            "duration_per_epoch_seconds": duration / config["epochs"],
            "memory_usage_gb": memory_usage,
            "gpu_memory_usage_gb": gpu_memory_usage,
        })
        
        wandb.finish()
