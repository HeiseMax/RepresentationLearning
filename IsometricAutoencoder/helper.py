import os
import torch


def load_optimizer_and_scheduler(filepath, optimizer, scheduler):
    """
    Loads the model's state from a checkpoint.
    Note: This only loads the model parameters. For full resume,
    use the resume_from_checkpoint in train_model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Optimizer and scheduler loaded from {filepath}")
    return optimizer, scheduler

def save_optimizer_and_scheduler(filepath, optimizer, scheduler):
    """
    Saves the optimizer and scheduler state to a checkpoint file.
    """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, filepath)
    
    print(f"Optimizer and scheduler saved to {filepath}")
