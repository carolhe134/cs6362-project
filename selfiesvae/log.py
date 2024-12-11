import torch
import os

def initialize_log_files(log_dir):
    log_files = {
        'training_loss': os.path.join(log_dir, 'training_loss.log'),
        'validation_loss': os.path.join(log_dir, 'validation_loss.log'),
        'kl_divergence': os.path.join(log_dir, 'kl_divergence.log'),
        'reconstruction_loss': os.path.join(log_dir, 'reconstruction_loss.log'),
    }
    for key, path in log_files.items():
        with open(path, 'w') as file:
            file.write(f"{key.capitalize()} Log\nEpoch\tValue\n")
    return log_files


def log_metrics(log_files, epoch, metrics):
    for key, value in metrics.items():
        with open(log_files[key], 'a') as file:
            file.write(f"{epoch}\t{value:.6f}\n")


def save_models(encoder, decoder, epoch, model_dir):
    epoch_dir = os.path.join(model_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(epoch_dir, 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(epoch_dir, 'decoder.pt'))