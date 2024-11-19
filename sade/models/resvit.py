from functools import partial

import numpy as np
import torch
import torch.nn as nn

from . import layers, layerspp, registry


@registry.register_model(name="resvit")
class ResidualViT(registry.BaseScoreModel):
    """
    Adapting model archtiecture from `simpler-diffusion` paper: https://arxiv.org/pdf/2410.19324
    The arch is like a unet with single (residual) skip connections

    f_down --> f_mid --- + --> f_up
           |-------------^

    TODO: name the module lists using module dict maybe so that prints are more readable
    """

    def __init__(
        self,
        config,
        spatial_dims: int = 2,
    ):
        super().__init__(config)

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.data = config.data
        self.init_filters = config.model.nf
        self.in_channels = self.data.num_channels
        self.out_channels = self.data.num_channels
        self.act = config.model.act

        # Configs for time embeddigns for the diffusion / noise scales
        self.time_embedding_sz = config.model.time_embedding_sz
        self.fourier_scale = config.model.fourier_scale
        self.learnable_embedding = config.model.learnable_embedding

        # Config related to Resnet Blocks
        # self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.blocks_down = config.model.blocks_down
        self.blocks_up = config.model.blocks_up
        self.channel_multipliers = config.model.channel_multipliers

        # Conv options will apply to all convolutions
        self.conv_size = config.model.conv_size

        self.compile = config.model.jit

        # Attention / Transformer related
        self.attention_heads = config.model.num_attention_heads

        # Setting up layer builders

        upsample_layer = partial(layerspp.get_upsample_layer, spatial_dims=spatial_dims)
        conv_layer = partial(
            layerspp.get_conv_layer,
            spatial_dims=spatial_dims,
            init_scale=config.model.init_scale,
            kernel_size=self.conv_size,
        )

        ResBlockpp = partial(
            layerspp.ResnetBlockBigGANpp,
            spatial_dims=spatial_dims,
            kernel_size=self.conv_size,
            act=self.act,
            init_scale=config.model.init_scale,
            temb_dim=self.time_embedding_sz * 2,  # sin and cos
        )

        # Initialize layers
        self.pool = layerspp.get_pooling_layer(spatial_dims=spatial_dims)
        self.init_conv = conv_layer(
            in_channels=self.in_channels, out_channels=self.init_filters
        )

        self.time_embed_layer = layerspp.make_time_cond_layers(
            self.time_embedding_sz,
            fourier_scale=self.fourier_scale,
            learnable_embedding=self.learnable_embedding,
        )

        base_channels = self.init_filters
        prev_mult = 1

        self.encoder = nn.ModuleList()

        for num_blocks_at_level, channel_mult in zip(
            self.blocks_down, self.channel_multipliers
        ):
            resblocks = nn.ModuleList()

            # Conv layer for channel_mult
            prev_channels = base_channels * prev_mult
            channels = base_channels * channel_mult
            preconv = conv_layer(in_channels=prev_channels, out_channels=channels)
            resblocks.append(preconv)

            for _ in range(num_blocks_at_level):
                resblocks.append(ResBlockpp(in_channels=channels))

            prev_mult = channel_mult
            self.encoder.append(resblocks)

        self.attention_block = layers.AttentionBlock(
            channels=channels, num_heads=self.attention_heads
        )

        self.decoder = nn.ModuleList()

        for num_blocks_at_level, channel_mult in zip(
            self.blocks_up, self.channel_multipliers[::-1]
        ):
            resblocks = nn.ModuleList()

            prev_channels = base_channels * prev_mult
            # These will be decreasing per level
            channels = base_channels * channel_mult
            upsample = upsample_layer(in_channels=prev_channels, out_channels=channels)
            resblocks.append(upsample)

            for _ in range(num_blocks_at_level):
                resblocks.append(ResBlockpp(in_channels=channels))

            prev_mult = channel_mult
            self.decoder.append(resblocks)

        self.out_conv = conv_layer(
            in_channels=self.init_filters, out_channels=self.out_channels
        )

    def forward(self, x, t_sigmas):

        t_emb = torch.log(t_sigmas)
        t_emb = self.time_embed_layer(t_emb)
        x = self.init_conv(x)

        down_x = []
        for i, enc_block in enumerate(self.encoder):
            preconv, *blocks = enc_block
            x = preconv(x)
            for res_block in blocks:
                x = res_block(x, t_emb)
            down_x.append(x)

            # End the block with a average pooling layer
            x = self.pool(x)
            # print(f"Computed down-layer {i}: {x.shape}")

        # Bottleneck block is tranformer-like
        x = self.attention_block(x) - x

        for i, dec_block in enumerate(self.decoder):
            upsample, *blocks = dec_block
            x = upsample(x)
            # print(f"Computed up-sample {i}: {x.shape} - down_x: {down_x[-1].shape}")

            x = (x + down_x.pop()) / np.sqrt(2)
            for res_block in blocks:
                x = res_block(x, t_emb)
            # print(f"Computed up-layer {i}: {x.shape}")

        x = self.out_conv(x)

        t_sigmas = t_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        x = x / t_sigmas

        return x
