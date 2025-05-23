


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.fc(z)

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, data_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.data_dim = data_dim
        self.latent_dim = latent_dim
    
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
    
    def get_loss(self, x):
        # Compute the reconstruction loss
        x_reconstructed = self.forward(x)
        loss = nn.MSELoss()(x_reconstructed, x)
        return loss
    
    def train(self, data, epochs=1000, batch_size=64, learning_rate=0.001):        
        # Define loss function and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size]

                # Compute loss
                loss = self.get_loss(batch_data)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
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