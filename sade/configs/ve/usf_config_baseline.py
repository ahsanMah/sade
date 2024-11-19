import os

from sade.configs.default_brain_configs import get_default_configs


def get_config():
   
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = True
    training.likelihood_weighting = False
    training.reduce_mean = True
    training.batch_size = 32
    training.n_iters = 500_000
    training.log_freq = 100
    training.eval_freq = 500
    training.snapshot_freq_for_preemption = 1000
    training.sampling_freq = 10_000
    training.load_pretrain = False
    training.pretrain_dir = "workdir/test/pretrain/"

    data = config.data
    data.dataset = "usf"
    data.image_size = (256, 256)
    data.num_channels = 3
    data.cache_rate = 0.0
    data.spatial_dims = 2
    data.dir_path = "/work2/jprieto/data/us-famli/save_frame/"
    data.splits_dir = "/ASD/ahsan_projects/Developer/braintypicality-scripts/split-keys"

    evaluate = config.eval
    evaluate.sample_size = 8
    evaluate.batch_size = 64

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.warmup = 1000
    optim.scheduler = "skip"

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.probability_flow = False
    sampling.snr = 0.17
    sampling.n_steps_each = 1
    sampling.noise_removal = True

    # model
    model = config.model
    # score-matching SDE params
    model.sigma_max = 633.0  # For sz=256
    model.sigma_min = 0.50
    model.name = "resvit"
    model.ema_rate = 0.9999
    # Noise conditioning related
    model.time_embedding_sz = 64
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = True
    model.num_scales = 1000
    # Conv-kernels
    model.conv_size = 3
    model.init_scale = 1.0
    model.act = "memswish"
    # enc-dec-blocks related
    model.nf = 32
    model.blocks_down = (1, 2, 2)
    model.blocks_up = (2, 2, 1)
    model.channel_multipliers = [1, 2, 4]
    model.num_attention_heads = 2
    model.dropout = 0.0

    return config
