import torch
from torch.autograd.functional import jvp, vjp


def isometric_loss(func, z, epsilon=1e-8):
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

def scaled_isometric_loss(func, z, eta=0.2, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    z_perm = z[torch.randperm(bs)] #?
    alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z) #?
    z_augmented = alpha*z + (1-alpha)*z_perm #?
    # z_augmented = z
    v = torch.randn(z.size()).to(z)
    Jv = jvp(
        func, z_augmented, v=v, create_graph=create_graph)[1]
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
    JTJv = (vjp(
        func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return TrG2/TrG**2

def conformal_loss(f: callable, z: torch.Tensor) -> torch.Tensor:
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

def generalized_conformal_loss(f, x):
    """
    Conformality loss for any f: R^n -> R^m (n < m), encouraging angle preservation.
    
    Args:
        f: differentiable function from (N, in_dim) -> (N, out_dim)
        x: input tensor of shape (N, in_dim)
        
    Returns:
        scalar loss
    """
    x = x.requires_grad_(True)
    fx = f(x)  # shape (N, out_dim)
    N, in_dim = x.shape
    out_dim = fx.shape[1]

    # Compute Jacobian: for each output dim, compute gradient wrt x
    grads = []
    for i in range(out_dim):
        grad = torch.autograd.grad(fx[:, i], x, grad_outputs=torch.ones_like(fx[:, i]), create_graph=True)[0]
        grads.append(grad)  # list of (N, in_dim)
    
    # Stack into (N, out_dim, in_dim) Jacobian tensor
    J = torch.stack(grads, dim=1)

    # Compute Gram matrix G = J^T @ J (per sample): shape (N, in_dim, in_dim)
    J_T = J.transpose(1, 2)
    G = torch.bmm(J_T, J)

    # Target: G ≈ s² * I → we normalize and minimize deviation from scaled identity
    eye = torch.eye(in_dim, device=x.device).unsqueeze(0)  # shape (1, in_dim, in_dim)
    
    # Normalize each Gram matrix by its trace
    trace = torch.trace(G) if in_dim == 1 else G.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    G_normalized = G / (trace + 1e-8)

    loss = torch.mean((G_normalized - eye) ** 2)
    return loss

def conformal_angle_loss(self, f, z):
    z_perm = z[torch.randperm(z.size(0))]
    
    x = f(z)
    x_perm = f(z_perm)

    cos_angle_z = torch.nn.functional.cosine_similarity(z, z_perm)
    cos_angle_x = torch.nn.functional.cosine_similarity(x, x_perm)

    loss = torch.mean((cos_angle_z - cos_angle_x) ** 2)
    return loss
