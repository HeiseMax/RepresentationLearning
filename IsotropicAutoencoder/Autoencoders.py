import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd.functional import jvp
from torch.autograd.functional import jacobian


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
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
        z = self.encoder(x)
        
        # Decode the latent representation
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed
    
    def encode(self, x):
        # Encode the input data to get the latent representation
        with torch.no_grad():
            z = self.encoder(x)
        return z
    
    def decode(self, z):
        # Decode the latent representation to get the reconstructed data
        with torch.no_grad():
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
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {loss.item():.4f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

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
    
    def get_default_scheduler(self, optimizer, scheduler_kwargs={"step_size":100, "gamma":0.1}):
        return StepLR(optimizer, **scheduler_kwargs)

    
    # train loop
    def train_model(self, data, val_data=None, epochs=1000, batch_size=64, learning_rate=0.001, optimizer=None, scheduler=None, optimizer_kwargs={}, scheduler_kwargs={"step_size":100, "gamma":0.1}, log_every=100, val_every=100, verbose=True):        
        # Define optimizer and scheduler
        if optimizer is None:
            optimizer = self.get_default_optimizer(learning_rate, optimizer_kwargs)
        if scheduler is None:
            scheduler = self.get_default_scheduler(optimizer, scheduler_kwargs)
        
        # Define data loaders
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        if val_data is not None:
            val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        else:
            val_dataloader = None
        
        # Training loop
        for epoch in range(epochs):
            self.epochs_trained += 1
            self.model.train()
            loss_list = []
            metrics_list = []
            for batch_data in dataloader:
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
                with torch.no_grad():
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



class IsometricAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, data_dim, latent_dim, lambda_iso=0.1):
        super(IsometricAutoencoder, self).__init__(encoder, decoder, data_dim, latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.lambda_iso = lambda_iso
    
    def get_loss(self, x):
        z = self.encoder(x)

        distortion = self.isometry_loss(self.decoder, z)

        x_reconstructed = self.decoder(z)
        mse = nn.MSELoss()(x_reconstructed, x)

        loss = mse + self.lambda_iso * distortion
        return loss
    
    def isometry_loss(self, func, z, epsilon=1e-8):
        """
        z: (batch_size, latent_dim) latent vectors sampled from Piso
        """
        
        # Sample u ~ Uniform(S^{d-1}), i.e., unit vector on sphere
        u = torch.randn_like(z)
        u = u / (u.norm(dim=1, keepdim=True) + epsilon)

        # Compute Jv = df(z) @ u
        Jv = jvp(func, z, u, create_graph=True)[1]

        # Compute norm of Jv and apply the isometric loss
        Jv_norm = Jv.norm(dim=1)
        loss = ((Jv_norm - 1.0) ** 2).mean()
        
        return loss
    
    
class ScaledIsometricAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, data_dim, latent_dim, lambda_iso=0.1):
        super(ScaledIsometricAutoencoder, self).__init__(encoder, decoder, data_dim, latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.lambda_iso = lambda_iso
    
    def get_loss(self, x):
        z = self.encoder(x)

        distortion = self.relaxed_distortion_measure(self.decoder, z)

        x_reconstructed = self.decoder(z)
        mse = nn.MSELoss()(x_reconstructed, x)

        loss = mse + self.lambda_iso * distortion
        return loss
    
    def relaxed_distortion_measure(self, func, z, eta=0.2, create_graph=True):
        '''
        func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
        '''
        bs = len(z)
        z_perm = z[torch.randperm(bs)] #?
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z) #?
        z_augmented = alpha*z + (1-alpha)*z_perm #?
        # z_augmented = z
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(
            func, z_augmented, v=v, create_graph=create_graph)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(
            func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()
        return TrG2/TrG**2


class ConformalAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, data_dim, latent_dim, lambda_iso=0.1):
        super(ConformalAutoencoder, self).__init__(encoder, decoder, data_dim, latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.lambda_iso = lambda_iso
    
    def get_loss(self, x):
        z = self.encoder(x)

        distortion = self.conformal_loss(self.decoder, z)

        x_reconstructed = self.decoder(z)
        mse = nn.MSELoss()(x_reconstructed, x)

        loss = mse + self.lambda_iso * distortion
        return loss
    
    def conformal_loss(self, f: callable, z: torch.Tensor) -> torch.Tensor:
        """
        Differentiable conformal loss measuring deviation from angle-preserving properties.
        
        Args:
            f: Differentiable function (typically neural network)
            z: Input tensor (must have requires_grad=True)
            
        Returns:
            Scalar loss value (lower means more conformal)
        """
        # Compute Jacobian matrix
        # J = torch.autograd.functional.jacobian(f, z, create_graph=True)
        B, in_dim = z.shape
        z = z.clone().detach().requires_grad_(True)

        y = f(z)  # (B, out_dim)
        out_dim = y.shape[1]

        jacobian = []

        for i in range(out_dim):
            grads = torch.autograd.grad(
                y[:, i].sum(), z, create_graph=True, retain_graph=True
            )[0]  # shape (B, in_dim)
            jacobian.append(grads.unsqueeze(1))  # shape (B, 1, in_dim)

        J = torch.cat(jacobian, dim=1)
        
        # Handle batch dimensions and reshape for matrix multiplication
        if z.dim() > 1:  # Batched inputs
            J = J.flatten(start_dim=1, end_dim=-2)
        
        # Compute J^T J and expected conformal scaling
        JTJ = torch.bmm(J.transpose(1,2), J)
        n = JTJ.size(1)
        trace = JTJ.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        lambda_scalar = trace / n
        
        # Create identity matrix matching device/dtype
        I = torch.eye(n, device=z.device, dtype=z.dtype)
        I = I.reshape((1, n, n))
        I = I.repeat(JTJ.shape[0], 1, 1)
        
        # Calculate Frobenius norm of deviation from conformal condition
        # reshape lambda_scalar to (batch_size, 2,2)
        lambda_scalar = lambda_scalar.reshape((-1, 1, 1))
        lambda_scalar = lambda_scalar.repeat(1, n, n)
        loss = torch.norm(JTJ - lambda_scalar * I, p='fro')**2
        
        return loss
    
    # def generalized_conformality_loss(self, f, x):
    #     """
    #     Conformality loss for any f: R^n -> R^m (n < m), encouraging angle preservation.
        
    #     Args:
    #         f: differentiable function from (N, in_dim) -> (N, out_dim)
    #         x: input tensor of shape (N, in_dim)
            
    #     Returns:
    #         scalar loss
    #     """
    #     x = x.requires_grad_(True)
    #     fx = f(x)  # shape (N, out_dim)
    #     N, in_dim = x.shape
    #     out_dim = fx.shape[1]

    #     # Compute Jacobian: for each output dim, compute gradient wrt x
    #     grads = []
    #     for i in range(out_dim):
    #         grad = torch.autograd.grad(fx[:, i], x, grad_outputs=torch.ones_like(fx[:, i]), create_graph=True)[0]
    #         grads.append(grad)  # list of (N, in_dim)
        
    #     # Stack into (N, out_dim, in_dim) Jacobian tensor
    #     J = torch.stack(grads, dim=1)

    #     # Compute Gram matrix G = J^T @ J (per sample): shape (N, in_dim, in_dim)
    #     J_T = J.transpose(1, 2)
    #     G = torch.bmm(J_T, J)

    #     # Target: G ≈ s² * I → we normalize and minimize deviation from scaled identity
    #     eye = torch.eye(in_dim, device=x.device).unsqueeze(0)  # shape (1, in_dim, in_dim)
        
    #     # Normalize each Gram matrix by its trace
    #     trace = torch.trace(G) if in_dim == 1 else G.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    #     G_normalized = G / (trace + 1e-8)

    #     loss = torch.mean((G_normalized - eye) ** 2)
    #     return loss