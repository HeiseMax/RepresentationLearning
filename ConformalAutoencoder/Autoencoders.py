import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd.functional import jvp
from torch.autograd.functional import jacobian

from metrics import isometry_loss, scaled_isometry_loss, conformality_trace_loss as conformality_loss


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.name = "Autoencoder"
        self.encoder = encoder
        self.decoder = decoder
        self.model = nn.Sequential(self.encoder, self.decoder)

        self.epochs_trained = 0
        self.loss_list = []
        self.val_loss_list = []
        self.metrics_list = {"reconstruction_loss": []}
        self.val_metrics_list = {"reconstruction_loss": []}
    
    def forward(self, x):
        # Encode the input data
        z = self.encode(x)
        
        # Decode the latent representation
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed
    
    def encode(self, x):
        # Encode the input data to get the latent representation
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        # Decode the latent representation to get the reconstructed data
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    

    # define for specialized autoencoders
    def get_metrics(self, x, val=False):
        # Compute all relevant metrics
        x_reconstructed = self.forward(x)
        loss = nn.MSELoss()(x_reconstructed, x)
        return [loss]
    
    def get_loss(self, metrics):
        # Combine metrics to compute loss
        return metrics[0]
    
    def get_batch_loss(self, loss_list):
        # Compute the average loss for the batch
        batch_loss = torch.mean(torch.tensor(loss_list))
        return batch_loss
    
    def get_batch_metrics(self, metrics):
        # Compute the average metrics for the batch
        batch_metrics = torch.mean(torch.tensor(metrics)[:,0]).item()
        return [batch_metrics]
    
    def log_loss_and_metrics(self, batch_loss, batch_metrics, epoch, epochs, val=False):
        # Log the loss and metrics for monitoring
        loss = batch_loss       
        if val:
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {loss.item():.8f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')

    def track_loss_and_metrics(self, batch_loss, batch_metrics, val=False):
        # Track and save the loss and metrics for monitoring
        loss = batch_loss
        if val:
            self.val_loss_list.append(loss.item())
            self.val_metrics_list["reconstruction_loss"].append(batch_metrics[0])
        else:
            self.loss_list.append(loss.item())
            self.metrics_list["reconstruction_loss"].append(batch_metrics[0])


    # default optimizer and scheduler
    def get_default_optimizer(self, learning_rate=0.001, optimizer_kwargs={}):
        return Adam(self.parameters(), lr=learning_rate, **optimizer_kwargs)
    
    def get_default_scheduler(self, optimizer, scheduler_kwargs={"step_size":100, "gamma":0.9}):
        return StepLR(optimizer, **scheduler_kwargs)

    
    # train loop
    def train_model(self, train_dataloader, val_dataloader=None, epochs=1000, batch_size=64, learning_rate=0.001, optimizer=None, scheduler=None, optimizer_kwargs={}, scheduler_kwargs={"step_size":100, "gamma":0.1}, log_every=100, val_every=100, verbose=True):        
        # Define optimizer and scheduler
        if optimizer is None:
            optimizer = self.get_default_optimizer(learning_rate, optimizer_kwargs)
        if scheduler is None:
            scheduler = self.get_default_scheduler(optimizer, scheduler_kwargs)
        
        # Training loop
        for epoch in range(epochs):
            self.epochs_trained += 1
            self.model.train()
            loss_list = []
            metrics_list = []
            for batch_data in train_dataloader:
                # Compute loss
                metrics = self.get_metrics(batch_data)
                metrics_list.append(metrics)
                loss = self.get_loss(metrics)
                loss_list.append(loss.item())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            batch_loss = self.get_batch_loss(loss_list)
            batch_metrics = self.get_batch_metrics(metrics_list)
            self.track_loss_and_metrics(batch_loss, batch_metrics, val=False)

            if (epoch + 1) % log_every == 0 and verbose:
                self.log_loss_and_metrics(batch_loss, batch_metrics, epoch, epochs)

            # Step the scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(batch_loss)
                else:
                    scheduler.step()

            # Validation step
            if val_dataloader is not None and (epoch + 1) % val_every == 0:
                self.model.eval()
                # with torch.no_grad():
                val_loss_list = []
                val_metrics_list = []
                for val_batch_data in val_dataloader:
                    val_metrics = self.get_metrics(val_batch_data, val=True)
                    val_metrics_list.append(val_metrics)
                    val_loss = self.get_loss(val_metrics)
                    val_loss_list.append(val_loss.item())

                val_batch_loss = self.get_batch_loss(val_loss_list)
                val_batch_metrics = self.get_batch_metrics(val_metrics_list)
                self.track_loss_and_metrics(val_batch_loss, val_batch_metrics, val=True)

                if verbose:
                    self.log_loss_and_metrics(val_batch_loss, val_batch_metrics, epoch, epochs, val=True)

        return optimizer, scheduler

    # Save and load model
    def save_checkpoint(self, filepath="checkpoint.pth"):
        """
        Saves the model's state, optimizer's state, current epoch,
        and custom class variables to a checkpoint file.
        """
        checkpoint = {
            'epochs_trained': self.epochs_trained,
            'model_state_dict': self.model.state_dict(),
            'loss_list': self.loss_list,
            'val_loss_list': self.val_loss_list,
            'metrics_list': self.metrics_list,
            'val_metrics_list': self.val_metrics_list,
            # Add any other class variables you want to save
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath} at epoch {self.epochs_trained + 1}")

    def load_model_from_checkpoint(self, filepath):
        """
        Loads the model's state from a checkpoint.
        Note: This only loads the model parameters. For full resume,
        use the resume_from_checkpoint in train_model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath)
        self.epochs_trained = checkpoint['epochs_trained']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_list = checkpoint['loss_list']
        self.val_loss_list = checkpoint['val_loss_list']
        self.metrics_list = checkpoint['metrics_list']
        self.val_metrics_list = checkpoint['val_metrics_list']
        print(f"Model and custom variables loaded from {filepath}")
        return self


class VariationalAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, latent_dim, beta=1.0):
        super(VariationalAutoencoder, self).__init__(encoder, decoder)
        self.name = "VariationalAutoencoder"
        self.latent_dim = latent_dim
        self.beta = beta

        self.metrics_list = {"reconstruction_loss": [], "kl_loss": []}
        self.val_metrics_list = {"reconstruction_loss": [], "kl_loss": []}

    def encode(self, x):
        # Encoder should output mean and logvar
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def get_metrics(self, x, val=False):
        x_reconstructed, mu, logvar = self.forward(x)
        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return [reconstruction_loss, kl_loss]

    def get_loss(self, metrics):
        reconstruction_loss, kl_loss = metrics
        return reconstruction_loss + self.beta * kl_loss

    def get_batch_loss(self, loss_list):
        batch_loss = torch.mean(torch.tensor(loss_list))
        return batch_loss

    def get_batch_metrics(self, metrics):
        batch_reconstruction_loss = torch.mean(torch.tensor(metrics)[:, 0]).item()
        batch_kl_loss = torch.mean(torch.tensor(metrics)[:, 1]).item()
        return [batch_reconstruction_loss, batch_kl_loss]

    def log_loss_and_metrics(self, batch_loss, batch_metrics, epoch, epochs, val=False):
        reconstruction_loss, kl_loss = batch_metrics
        if val:
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {batch_loss:.8f}, Reconstruction Loss: {reconstruction_loss:.8f}, KL Loss: {kl_loss:.8f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {batch_loss:.8f}, Reconstruction Loss: {reconstruction_loss:.8f}, KL Loss: {kl_loss:.8f}')

    def track_loss_and_metrics(self, batch_loss, batch_metrics, val=False):
        reconstruction_loss, kl_loss = batch_metrics
        if val:
            self.val_loss_list.append(batch_loss)
            self.val_metrics_list["reconstruction_loss"].append(reconstruction_loss)
            self.val_metrics_list["kl_loss"].append(kl_loss)
        else:
            self.loss_list.append(batch_loss)
            self.metrics_list["reconstruction_loss"].append(reconstruction_loss)
            self.metrics_list["kl_loss"].append(kl_loss)


class IsometricAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, lambda_iso=1.0):
        super(IsometricAutoencoder, self).__init__(encoder, decoder)
        self.name = "IsometricAutoencoder"
        self.lambda_iso = lambda_iso

        self.isometry_loss = isometry_loss

        self.metrics_list = {"reconstruction_loss": [], "isometric_loss": []}
        self.val_metrics_list = {"reconstruction_loss": [], "isometric_loss": []}
    
    def get_metrics(self, x, val=False):
        # Compute all relevant metrics
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        isometric_loss = self.isometry_loss(self.decoder, z)
        return [reconstruction_loss, isometric_loss]
    
    def get_loss(self, metrics):
        # Combine metrics to compute loss
        reconstruction_loss, isometric_loss = metrics
        return reconstruction_loss + self.lambda_iso * isometric_loss
    
    def get_batch_loss(self, loss_list):
        # Compute the average loss for the batch
        batch_loss = torch.mean(torch.tensor(loss_list))
        return batch_loss
    
    def get_batch_metrics(self, metrics):
        # Compute the average metrics for the batch
        batch_reconstruction_loss = torch.mean(torch.tensor(metrics)[:, 0]).item()
        batch_isometric_loss = torch.mean(torch.tensor(metrics)[:, 1]).item()
        return [batch_reconstruction_loss, batch_isometric_loss]
    
    def log_loss_and_metrics(self, batch_loss, batch_metrics, epoch, epochs, val=False):
        # Log the loss and metrics for monitoring
        reconstruction_loss, isometric_loss = batch_metrics
        if val:
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {batch_loss.item():.8f}, Reconstruction Loss: {reconstruction_loss:.8f}, Isometric Loss: {isometric_loss:.8f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {batch_loss.item():.8f}, Reconstruction Loss: {reconstruction_loss:.8f}, Isometric Loss: {isometric_loss:.8f}')

    def track_loss_and_metrics(self, batch_loss, batch_metrics, val=False):
        # Track and save the loss and metrics for monitoring
        reconstruction_loss, isometric_loss = batch_metrics
        if val:
            self.val_loss_list.append(batch_loss.item())
            self.val_metrics_list["reconstruction_loss"].append(reconstruction_loss)
            self.val_metrics_list["isometric_loss"].append(isometric_loss)
        else:
            self.loss_list.append(batch_loss.item())
            self.metrics_list["reconstruction_loss"].append(reconstruction_loss)
            self.metrics_list["isometric_loss"].append(isometric_loss)


class ScaledIsometricAutoencoder(IsometricAutoencoder):
    def __init__(self, encoder, decoder, lambda_iso=1.0):
        super(ScaledIsometricAutoencoder, self).__init__(encoder, decoder)
        self.name = "ScaledIsometricAutoencoder"
        self.lambda_iso = lambda_iso

        self.scaled_isometry_loss = scaled_isometry_loss

        self.metrics_list = {"reconstruction_loss": [], "isometric_loss": []}
        self.val_metrics_list = {"reconstruction_loss": [], "isometric_loss": []}
    
    def get_metrics(self, x, val=False):
        # Compute all relevant metrics
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        isometric_loss = self.scaled_isometry_loss(self.decoder, z)
        return [reconstruction_loss, isometric_loss]


class ConformalAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, lambda_conf=0.1):
        super(ConformalAutoencoder, self).__init__(encoder, decoder)
        self.name = "ConformalAutoencoder"
        self.lambda_conf = lambda_conf

        self.conformality_loss = conformality_loss

        self.metrics_list = {"reconstruction_loss": [], "conformal_loss": []}
        self.val_metrics_list = {"reconstruction_loss": [], "conformal_loss": []}

    def get_metrics(self, x, val=False):
        # Compute all relevant metrics
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        conformal_loss = self.conformality_loss(self.decoder, z)
        return [reconstruction_loss, conformal_loss]
    
    def get_loss(self, metrics):
        # Combine metrics to compute loss
        reconstruction_loss, conformal_loss = metrics
        return reconstruction_loss + self.lambda_conf * conformal_loss
    
    def get_batch_loss(self, loss_list):
        # Compute the average loss for the batch
        batch_loss = torch.mean(torch.tensor(loss_list))
        return batch_loss
    
    def get_batch_metrics(self, metrics):
        # Compute the average metrics for the batch
        batch_reconstruction_loss = torch.mean(torch.tensor(metrics)[:, 0]).item()
        batch_conformal_loss = torch.mean(torch.tensor(metrics)[:, 1]).item()
        return [batch_reconstruction_loss, batch_conformal_loss]
    
    def log_loss_and_metrics(self, batch_loss, batch_metrics, epoch, epochs, val=False):
        # Log the loss and metrics for monitoring
        reconstruction_loss, conformal_loss = batch_metrics
        if val:
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {batch_loss.item():.8f}, Reconstruction Loss: {reconstruction_loss:.8f}, Conformal Loss: {conformal_loss:.8f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {batch_loss.item():.8f}, Reconstruction Loss: {reconstruction_loss:.8f}, Conformal Loss: {conformal_loss:.8f}')

    def track_loss_and_metrics(self, batch_loss, batch_metrics, val=False):
        # Track and save the loss and metrics for monitoring
        reconstruction_loss, conformal_loss = batch_metrics
        if val:
            self.val_loss_list.append(batch_loss.item())
            self.val_metrics_list["reconstruction_loss"].append(reconstruction_loss)
            self.val_metrics_list["conformal_loss"].append(conformal_loss)
        else:
            self.loss_list.append(batch_loss.item())
            self.metrics_list["reconstruction_loss"].append(reconstruction_loss)
            self.metrics_list["conformal_loss"].append(conformal_loss)
