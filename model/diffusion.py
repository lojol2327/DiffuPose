import math
import torch

from torch import nn
import torch.nn.functional as F

from inspect import isfunction
from tqdm import tqdm
from einops import  reduce

from common.loss import get_w, post_processing


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    '''
    from timesteps, t is randomly sampled -> batch_size
    To match coefficient idx with randomly sampled t
    output shape -> (b, 1, 1, 1) for broadcasting
    '''
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Diffusion(nn.Module):
    
    def __init__(
        self,
        denoise_fn,
        *,
        n_frame,
        n_joint,
        channels_j = 3,
        channels_k = 2,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., 
        p2_loss_weight_k = 1
    ):
        super().__init__()

        self.channels_j = channels_j
        self.channels_k = channels_k
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.denoise_fn = denoise_fn
        self.objective = objective

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
    
    '----------------------------------Sampling----------------------------------'
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        model_output = self.denoise_fn(x, cond, t)
        
        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output[:,:,:,:3])
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape_j, cond):
        device = self.betas.device
        b = shape_j[0]
        
        joint = torch.randn(shape_j, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            joint = self.p_sample(joint, cond, torch.full((b,), i, device=device, dtype=torch.long))

        return joint

    @torch.no_grad()
    def sample(self, cond, batch_size = 16):
        n_frame = self.n_frame
        n_joint = self.n_joint
        
        channels_j = self.channels_j
        
        shape_j = (batch_size, n_frame, n_joint, channels_j)
        
        return self.p_sample_loop(shape_j, cond)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img
    '----------------------------------------------------------------------------'

    '----------------------------------Training----------------------------------'
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    def p_losses(self, x_start, cond, t, noise = None):
        b, f, j, c = x_start.shape
        noise_j = default(noise, lambda: torch.randn_like(x_start))

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise_j)
        model_out = self.denoise_fn(x_t, cond, t)
        
        # model_out = model_out[:,:,:,:3]
        
        if self.objective == 'pred_noise':
            target = torch.cat((noise_j, cond), -1)
            # target = noise_j
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss[:,:,:,3:] = loss[:,:,:,3:] * 0.8
        weight = get_w(weighted=True).cuda()
        weighted_loss = loss * weight
        
        weighted_loss = reduce(weighted_loss, 'b ... -> b (...)', 'mean')

        weighted_loss = weighted_loss * extract(self.p2_loss_weight, t, weighted_loss.shape)
        return weighted_loss.mean(), loss.mean()
    
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    '----------------------------------------------------------------------------'
    def forward(self, joint, cond, *args, **kwargs):
        b, f, j, c, device, n_frame, n_joint = *joint.shape, joint.device, self.n_frame, self.n_joint
        # assert f == n_frame and j == n_joint, f'input size must be {n_frame} x {n_joint}'
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(joint, cond, t, *args, **kwargs)
