import torch
from torch.autograd.functional import jvp, vjp


from torch.func import functional_call, jacrev, vmap

def gini(w: torch.Tensor) -> torch.Tensor:
    r"""The Gini coefficient from the `"Improving Molecular Graph Neural
    Network Explainability with Orthonormalization and Induced Sparsity"
    <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization penalty :math:`\in [0, 1]` for each row of a
    matrix according to

    .. math::
        \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
         - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

    and returns an average over all rows.

    Args:
        w (torch.Tensor): A two-dimensional tensor.
    """
    s = 0
    for row in w:
        t = row.repeat(row.size(0), 1)
        u = (t - t.T).abs().sum() / (2 * (row.size(-1)**2 - row.size(-1)) *
                                     row.abs().mean() + torch.finfo().eps)
        s += u
    s /= w.shape[0]
    return s

def get_batch_jacobian(func, inputs):
    params = dict(func.named_parameters())

    def fmodel(params, inputs):
        return functional_call(func, params, inputs.flatten().unsqueeze(0)).flatten()

    jacobians = vmap(jacrev(fmodel, argnums=(1)), in_dims=(None,0))(params, inputs)
    return jacobians

def evaluate_conformality(model, data):
    # still needs flattening for convolutions
    model.eval()
    with torch.no_grad():
        data_dim = data.shape[1]
        # latent samples
        latent = model.encode(data)
        latent_dim = latent.shape[1]

        # Compute the Jacobian of the decoder
        jacobians = get_batch_jacobian(model.decoder, latent).double()
        jTjs = torch.einsum('bji,bjk->bik', jacobians, jacobians)
        print(jTjs.type())
        traces = vmap(torch.trace)(jTjs)
        lambda_factors = traces / latent.shape[1]

        # gini for diagonal elements
        diagonals = jTjs.diagonal(dim1=1, dim2=2)
        gini_value = gini(diagonals)

        # off diag mean, norm
        off_diagonal = jTjs - torch.diag_embed(diagonals)
        off_diag_mean = off_diagonal.abs().mean(dim=(1, 2))
        off_diag_norm = off_diagonal.norm(dim=(1, 2))
        off_diag_mean_normed = off_diag_mean / lambda_factors
        off_diag_norm_normed = off_diag_norm / lambda_factors

        # jTj - lambda*I mean, norm
        eye = torch.eye(jTjs.shape[1], device=jTjs.device).repeat(jTjs.shape[0], 1, 1)
        jTj_minus_lambdaI = jTjs - lambda_factors.unsqueeze(1).unsqueeze(2).repeat(1, jTjs.shape[1], jTjs.shape[1]) * eye
        jTj_minus_lambdaI_mean = jTj_minus_lambdaI.abs().mean(dim=(1, 2))
        jTj_minus_lambdaI_norm = jTj_minus_lambdaI.norm(dim=(1, 2))
        jTj_minus_lambdaI_mean_normed = jTj_minus_lambdaI_mean / lambda_factors
        jTj_minus_lambdaI_norm_normed = jTj_minus_lambdaI_norm / lambda_factors

        # determinant of jacobian vs sqrt lambda**m
        jacobian_determinants = torch.sqrt(torch.linalg.det(jTjs))
        determinant_vs_estimate = jacobian_determinants / (torch.sqrt(lambda_factors**latent_dim) + torch.finfo(torch.float32).eps)

        jacobian_log_determinants = 0.5 * torch.logdet(jTjs)
        # print(lambda_factors)
        # print(torch.linalg.slogdet(jTjs)[0])
        # print(jacobian_log_determinants)
        log_det_estimate = 0.5 * latent_dim * torch.log(lambda_factors)
        log_determinant_vs_estimate = jacobian_log_determinants - log_det_estimate
        # print(log_det_estimate)

        # std of latent space
        latent_std = latent.std(dim=0)
        
        return {
            'reconstruction_error': torch.nn.MSELoss()(data, model.decode(latent)).item(),
            'diagonal_gini': gini_value.item(),
            'lambda_mean': lambda_factors.mean().item(),
            'lambda_std': lambda_factors.std().item(),
            'lambda_std_normed': (lambda_factors.std() /lambda_factors.mean()).item(),
            'off_diag_mean': off_diag_mean.mean().item(),
            'off_diag_norm': off_diag_norm.mean().item(),
            'off_diag_mean_normed': off_diag_mean_normed.mean().item(),
            'off_diag_norm_normed': off_diag_norm_normed.mean().item(),
            'jTj_minus_lambdaI_mean': jTj_minus_lambdaI_mean.mean().item(),
            'jTj_minus_lambdaI_norm': jTj_minus_lambdaI_norm.mean().item(),
            'jTj_minus_lambdaI_mean_normed': jTj_minus_lambdaI_mean_normed.mean().item(),
            'jTj_minus_lambdaI_norm_normed': jTj_minus_lambdaI_norm_normed.mean().item(),
            'determinant_vs_estimate_mean': determinant_vs_estimate.mean().item(),
            'determinant_vs_estimate_std': determinant_vs_estimate.std().item(),
            'log_determinant_vs_estimate_mean': log_determinant_vs_estimate.mean().item(),
            'log_determinant_vs_estimate_std': log_determinant_vs_estimate.std().item(),
            'latent_std': latent_std.mean().item(),
            'latent_std_max': latent_std.max().item(),
            'latent_std_min': latent_std.min().item(),
            'latent_norm': latent.norm().item(),
        }
    
def isometry_loss(func, z, epsilon=1e-8, eta=0.2, create_graph=True, augment=True):
    """
    z: (batch_size, latent_dim) latent vectors sampled from Piso
    """
    bs = len(z)
    if augment:
        z_perm = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z
    
    # Sample u ~ Uniform(S^{d-1}), i.e., unit vector on sphere
    u = torch.randn_like(z_augmented)
    u = u / (u.norm(dim=1, keepdim=True) + epsilon)

    # Compute Jv = df(z) @ u
    Jv = jvp(func, z_augmented, u, create_graph=create_graph)[1]

    # Compute norm of Jv and apply the isometric loss
    Jv_norm = Jv.norm(dim=1)
    loss = ((Jv_norm - 1.0) ** 2).mean()

    return loss

def scaled_isometry_loss(func, z, eta=0.2, create_graph=True, augment=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    if augment:
        z_perm = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z

    v = torch.randn(z.size()).to(z)
    Jv = jvp(
        func, z_augmented, v=v, create_graph=create_graph)[1]
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean() #TODO this looks wrong? Use JtJv instead of Jv^2?
    JTJv = (vjp(
        func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return TrG2/TrG**2



def conformality_trace_loss(func, z, eta=0.2, create_graph=True, augment=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    if augment:
        z_perm = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z

    v = torch.randn(z.size()).to(z)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-7)
    Jv = jvp(
        func, z_augmented, v=v, create_graph=create_graph)[1]
    JTJv = (vjp(
        func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1)
    TrG2 = torch.sum(JTJv**2, dim=1)
    m = z.shape[1]
    TrN = (TrG2 - (2/m *(TrG**2)))
    # TrN = TrG2 / (TrG**2) #needs gradient clipping

    return (TrN**2).mean()


def conformality_cosine_loss(f, z, lam=1):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)
    u = torch.randn_like(z)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-8)

    def make_orthogonal(a):
        """
        a: Tensor of shape (batch_size, dim)
        Returns: Tensor of shape (batch_size, dim), each vector orthogonal to corresponding input
        """
        # Normalize input vectors (shape: [B, D])
        a_norm = a / a.norm(dim=1, keepdim=True)

        # Random vectors (same shape as input)
        b = torch.randn_like(a)

        # Dot product per batch (shape: [B])
        proj_coeff = torch.sum(a_norm * b, dim=1, keepdim=True)

        # Remove projection of b onto a => b_orth = b - proj_a(b)
        b_orth = b - proj_coeff * a_norm

        return b_orth
    
    v = make_orthogonal(u)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)


    Jv = jvp(f, z, v, create_graph=True)[1]
    Ju = jvp(f, z, u, create_graph=True)[1]

    # Compute the angle between the two vectors
    cos_angle = torch.cosine_similarity(Ju, Jv, dim=1)

    angle_loss = torch.mean((cos_angle) ** 2)
    return angle_loss

def conformality_cosine2_loss(f, z, augment=True, eta=0.2):
    #sample batchsize pairs of orthogonal unit vectors
    bs = len(z)
    if augment:
        z_perm = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z
    u = torch.randn_like(z)
    v = torch.randn_like(z)

    cos_uv = torch.cosine_similarity(u,v, dim=1)


    Jv = jvp(f, z_augmented, v, create_graph=True)[1].view(bs, -1)
    Ju = jvp(f, z_augmented, u, create_graph=True)[1].view(bs, -1)

    # Compute the angle between the two vectors
    cos_JuJv = torch.cosine_similarity(Ju, Jv, dim=1)

    angle_loss = torch.mean((cos_uv - cos_JuJv) ** 2)
    return angle_loss


def regularization1(func, z, create_graph=True, augment=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)

    v = torch.randn(z.size()).to(z)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-7)
    Jv = jvp(
        func, z, v=v, create_graph=create_graph)[1]
    
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1)
    m = z.shape[1]

    return ((TrG.mean() / m) -1)**2

def regularization5(func, z, goal_norm=40.0, create_graph=True, augment=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)

    return (z.norm() - goal_norm)**2

def regularization4(f, z, lam=1):
    # sample pairs pairs of orthogonal unit vectors, compare cosine similarity and length
    bs = len(z)
    v = torch.randn_like(z)

    Jv = jvp(f, z, v, create_graph=True)[1]

    v_len = v.norm(dim=1)
    Jv_len = Jv.norm(dim=1)

    return ((v_len/Jv_len).mean() -1)**2
