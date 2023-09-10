"""All functions and modules related to model definition.
"""

import logging

import numpy as np
import torch
from torchinfo import summary

import sade.sde_lib as sde_lib
import wandb

_MODELS = {}


class BaseScoreModel(torch.nn.Module):
    """Base class for score models."""
    
    def __init__(self, config):
        super().__init__()
        self.model_config = config.model
        self.sde = create_sde(config)
        self.register_buffer("sigmas", self.sde.discrete_sigmas)

def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )

    return sigmas


def create_model(config, log_grads=False, print_summary=False):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)

    if print_summary:
        logging.info(score_model)
        summary(
            score_model,
            input_data=(
                torch.zeros(
                    size=(1, config.data.num_channels, *config.data.image_size)
                ),
                torch.zeros(
                    1,
                ),
            ),
        )

    if log_grads:
        wandb.watch(score_model, log="all", log_freq=config.training.sampling_freq)

    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model

def create_sde(config):
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    return sde

def get_model_fn(model, train=False, amp=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """

        with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
            if not train:
                model.eval().requires_grad_(False)
                return model(x, labels)
            else:
                model.train()
                return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False, amp=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train, amp=amp)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            labels = t * 999
            score = model_fn(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = -score / sde._unsqueeze(std)
            return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t):
            labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


@torch.inference_mode()
def denoise_update(x, eps=1e-2):
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * DENOISE_EPS
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x


def build_score_norm_fn(config, score_model, return_norm=True, denoise=False):
    """Build a function to compute the norm of the score function."""

    # Setup SDEs
    if config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
    else:
        raise NotImplementedError(
            f"SDE {config.training.sde} has not been tested with MSMA."
        )

    score_fn = get_score_fn(
        sde,
        score_model,
        train=False,
        continuous=config.training.continuous,
        amp=config.training.use_fp16,
    )

    n_timesteps = config.model.num_scales
    eps = config.msma.min_timestep
    end = config.msma.max_timestep

    schedule = config.msma.schedule if "schedule" in config.msma else "geometric"
    if schedule == "linear":
        logging.info("Using linearly spaced sigmas.")
        msma_sigmas = torch.linspace(
            max(1e-1, sde.sigma_min),
            sde.sigma_max,
            config.msma.n_timesteps,
            device=config.device,
        )
        timesteps = sde.noise_schedule_inverse(msma_sigmas).to(config.device)
    elif schedule == "geometric":
        logging.info("Using geometrically spaced sigmas.")
        msma_sigmas = torch.exp(
            torch.linspace(
                np.log(max(1e-1, sde.sigma_min)),
                np.log(sde.sigma_max),
                config.msma.n_timesteps,
                device=config.device,
            )
        )
        timesteps = sde.noise_schedule_inverse(msma_sigmas).to(config.device)
    else:
        logging.warning("Using linear spaced timesteps to determine sigmas.")
        timesteps = torch.linspace(eps, end, n_timesteps, device=config.device)
        timesteps = timesteps[:: n_timesteps // config.msma.n_timesteps]

    def scorer(x):
        """Compute scores for a batch of samples.
        Indexing into the timesteps list grabs the *exact* sigmas used during training
        The alternate would be to compute a linearly spaced list of sigmas of size msma.n_timesteps
        However, this would technicaly output sigmas never seen during training...
        """

        n_timesteps = len(timesteps)

        with torch.no_grad():
            if denoise:
                x = denoise_update(x)

            if return_norm:
                scores = torch.zeros(
                    (x.shape[0], n_timesteps), device=config.device, dtype=torch.float32
                )
            else:
                scores = torch.zeros(
                    (x.shape[0], n_timesteps, *x.shape[2:]),
                    device=config.device,
                    dtype=torch.float32,
                )

            for i, tidx in enumerate(range(0, n_timesteps)):
                # logging.info(f"sigma {i}")
                t = timesteps[tidx]
                vec_t = torch.ones(x.shape[0], device=config.device) * t
                std = sde.marginal_prob(torch.zeros_like(x), vec_t)[1]
                score = score_fn(x, vec_t)

                if return_norm:
                    score = (
                        torch.linalg.norm(
                            score.reshape((score.shape[0], -1)),
                            dim=-1,
                        )
                        * std
                    )
                else:
                    score = (score * std[:, None, None, None, None]).sum(dim=1)

                scores[:, i, ...].copy_(score)

        return scores

    return scorer
