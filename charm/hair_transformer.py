from __future__ import annotations

from math import ceil
from functools import partial
from tqdm import tqdm
from scipy.interpolate import CubicSpline

from beartype.typing import Tuple, Callable, List

import numpy as np
from einops import rearrange, repeat, reduce, pack
from gateloop_transformer import SimpleGateLoopLayer

import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from x_transformers import Decoder
from x_transformers.x_transformers import LayerIntermediates
from x_transformers.autoregressive_wrapper import eval_decorator
from .michelangelo import ShapeConditioner

from .utils import (
    piecewise_discretize,
    piecewise_undiscretize,
    set_module_requires_grad_,
    default,
    exists,
    is_tensor_empty,
    safe_cat,
    identity,
)
from .utils.typing import Float, Bool, typecheck


class FiLM(Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_gamma = nn.Linear(dim, dim_out, bias = False)
        self.to_beta = nn.Linear(dim, dim_out)

        self.gamma_mult = nn.Parameter(torch.zeros(1,))
        self.beta_mult = nn.Parameter(torch.zeros(1,))

    def forward(self, x, cond):
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = tuple(rearrange(t, 'b d -> b 1 d') for t in (gamma, beta))

        # for initializing to identity

        gamma = (1 + self.gamma_mult * gamma.tanh())
        beta = beta.tanh() * self.beta_mult

        # classic film

        return x * gamma + beta

class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    ):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim, use_heinsen = use_heinsen)
            self.gateloops.append(gateloop)

    def forward(
        self,
        x,
        cache = None
    ):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        return x, new_caches


def top_k_2(logits, frac_num_tokens=0.1, k=None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


class HairTransformerDiscrete(Module):
    @typecheck
    def __init__(
        self,
        *,
        num_discrete_translation_x = (24, 80, 24),
        continuous_range_translation_x: List[float, float] = [-0.5, -0.1, 0.1, 0.5],
        num_discrete_translation_y = (24, 40, 64),
        continuous_range_translation_y: List[float, float] = [-0.5, 0.0, 0.3, 0.5],
        num_discrete_translation_z = (24, 80, 24),
        continuous_range_translation_z: List[float, float] = [-0.5, -0.15, 0.1, 0.5],
        dim_translation_embed = 64,
        num_discrete_width = (64, 64),
        continuous_range_width: List[float, float] = [0, 0.03, 0.1],
        dim_width_embed = 64,
        num_discrete_thickness = (64, 64),
        continuous_range_thickness: List[float, float] = [0, 0.02, 0.1],
        dim_thickness_embed = 64,
        embed_order = 'ctrs',
        dim: int | Tuple[int, int] = 512,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            attn_num_mem_kv = 4
        ),
        max_token_len = 8192,
        dropout = 0.,
        coarse_pre_gateloop_depth = 2,
        coarse_post_gateloop_depth = 0,
        coarse_adaptive_rmsnorm = False,
        gateloop_use_heinsen = False,
        pad_id = -1,
        split_id = -2,
        num_sos_tokens = None,
        condition_on_shape = True,
        shape_cond_with_cross_attn = False,
        shape_cond_with_film = False,
        shape_cond_with_cat = False,
        shape_condition_model_type = '',
        shape_condition_len = 1,
        shape_condition_dim = None,
        cross_attn_num_mem_kv = 4,
        loss_weight: dict = dict(
            eos = 1.0,
            mos = 1.0,
            translation = 1.0,
            width = 1.0,
            thickness = 1.0,
            eos_ce_weight = 2000.0,
            mos_ce_weight = 40.0,
        ),
    ):
        super().__init__()

        assert np.sum(num_discrete_translation_x) == np.sum(num_discrete_translation_y) == np.sum(num_discrete_translation_z)
        self.num_discrete_translation = np.sum(num_discrete_translation_x)
        self.discretize_translation_x = partial(piecewise_discretize, num_discrete=num_discrete_translation_x, continuous_range=continuous_range_translation_x)
        self.discretize_translation_y = partial(piecewise_discretize, num_discrete=num_discrete_translation_y, continuous_range=continuous_range_translation_y)
        self.discretize_translation_z = partial(piecewise_discretize, num_discrete=num_discrete_translation_z, continuous_range=continuous_range_translation_z)
        self.undiscretize_translation_x = partial(piecewise_undiscretize, num_discrete=num_discrete_translation_x, continuous_range=continuous_range_translation_x)
        self.undiscretize_translation_y = partial(piecewise_undiscretize, num_discrete=num_discrete_translation_y, continuous_range=continuous_range_translation_y)
        self.undiscretize_translation_z = partial(piecewise_undiscretize, num_discrete=num_discrete_translation_z, continuous_range=continuous_range_translation_z)
        self.translation_embed = nn.Embedding(self.num_discrete_translation, dim_translation_embed)

        self.num_discrete_width = np.sum(num_discrete_width)
        self.continuous_range_width = continuous_range_width
        self.discretize_width = partial(piecewise_discretize, num_discrete=num_discrete_width, continuous_range=continuous_range_width)
        self.undiscretize_width = partial(piecewise_undiscretize, num_discrete=num_discrete_width, continuous_range=continuous_range_width)
        self.width_embed = nn.Embedding(self.num_discrete_width, dim_width_embed)

        self.num_discrete_thickness = np.sum(num_discrete_thickness)
        self.continuous_range_thickness = continuous_range_thickness
        self.discretize_thickness = partial(piecewise_discretize, num_discrete=num_discrete_thickness, continuous_range=continuous_range_thickness)
        self.undiscretize_thickness = partial(piecewise_undiscretize, num_discrete=num_discrete_thickness, continuous_range=continuous_range_thickness)
        self.thickness_embed = nn.Embedding(self.num_discrete_thickness, dim_thickness_embed)

        self.embed_order = embed_order

        self.dim = dim
        init_dim = 3 * dim_translation_embed + dim_width_embed + dim_thickness_embed
        self.project_in = nn.Linear(init_dim, dim)

        num_sos_tokens = default(num_sos_tokens, 1 if not condition_on_shape or not shape_cond_with_film else 4)
        assert num_sos_tokens > 0

        self.num_sos_tokens = num_sos_tokens
        self.sos_token = nn.Parameter(torch.randn(num_sos_tokens, dim))
        self.mos_token = nn.Parameter(torch.randn(1, dim))
        self.eos_token = nn.Parameter(torch.randn(1, dim))

        self.max_seq_len = max_token_len

        self.condition_on_shape = condition_on_shape
        self.shape_cond_with_cross_attn = False
        self.shape_cond_with_cat = False
        self.shape_condition_model_type = ''
        self.conditioner = None
        dim_shape = None

        if condition_on_shape:
            assert shape_cond_with_cross_attn or shape_cond_with_film or shape_cond_with_cat
            self.shape_cond_with_cross_attn = shape_cond_with_cross_attn
            self.shape_cond_with_cat = shape_cond_with_cat
            self.shape_condition_model_type = shape_condition_model_type

            self.conditioner = ShapeConditioner(dim_latent=shape_condition_dim)
            self.to_cond_dim = nn.Linear(self.conditioner.dim_model_out * 2, dim)
            self.to_cond_dim_head = nn.Linear(self.conditioner.dim_model_out, dim)


            dim_shape = self.conditioner.dim_latent
            set_module_requires_grad_(self.conditioner, False)

            self.shape_coarse_film_cond = FiLM(dim_shape, dim) if shape_cond_with_film else identity

        self.coarse_gateloop_block = GateLoopBlock(dim, depth=coarse_pre_gateloop_depth, use_heinsen=gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        self.coarse_post_gateloop_block = GateLoopBlock(dim, depth=coarse_post_gateloop_depth, use_heinsen=gateloop_use_heinsen) if coarse_post_gateloop_depth > 0 else None

        self.coarse_adaptive_rmsnorm = coarse_adaptive_rmsnorm

        self.decoder = Decoder(
            dim=dim,
            depth=attn_depth,
            heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_flash=flash_attn,
            attn_dropout=dropout,
            ff_dropout=dropout,
            use_adaptive_rmsnorm=coarse_adaptive_rmsnorm,
            dim_condition=dim_shape,
            cross_attend=self.shape_cond_with_cross_attn,
            cross_attn_dim_context=dim_shape,
            cross_attn_num_mem_kv=cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # to logits
        self.to_mos_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.to_eos_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.to_translation_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3 * self.num_discrete_translation)
        )
        self.to_width_logits = nn.Sequential(
            nn.Linear(dim + 3 * dim_translation_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, self.num_discrete_width)
        )
        self.to_thickness_logits = nn.Sequential(
            nn.Linear(dim + 3 * dim_translation_embed + dim_width_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, self.num_discrete_thickness)
        )

        # padding id
        self.pad_id = pad_id
        self.split_id = split_id

        # loss weight
        self.loss_weight = loss_weight


    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    @torch.no_grad()
    def embed_pc(self, pc: Tensor):
        pc_head, pc_embed = self.conditioner(shape=pc)
        pc_embed = torch.cat([self.to_cond_dim_head(pc_head), self.to_cond_dim(pc_embed)], dim=-2).detach()
        return pc_embed


    @typecheck
    def recon_hair(
        self,
        translation_logits: Float['b np 3 nd'],
        width_logits: Float['b np nd'],
        thickness_logits: Float['b np nd'],
        hair_mask: Bool['b np']
    ):
        translation_argmax = translation_logits.argmax(dim=-1)
        recon_translation = torch.stack([
            self.undiscretize_translation_x(translation_argmax[:, :, 0]),
            self.undiscretize_translation_y(translation_argmax[:, :, 1]),
            self.undiscretize_translation_z(translation_argmax[:, :, 2])
        ], dim=-1)

        recon_translation = recon_translation.masked_fill(~hair_mask.unsqueeze(-1), float('nan'))
        recon_width = self.undiscretize_width(width_logits.argmax(dim=-1))
        recon_width = recon_width.masked_fill(~hair_mask, float('nan'))
        recon_thickness = self.undiscretize_thickness(thickness_logits.argmax(dim=-1))
        recon_thickness = recon_thickness.masked_fill(~hair_mask, float('nan'))

        return {
            'translation': recon_translation,
            'width': recon_width,
            'thickness': recon_thickness
        }


    @typecheck
    def sample_hair(
        self,
        translation: Float['b np 3 nd'],
        width: Float['b np nd'],
        thickness: Float['b np nd'],
        next_mos: Float['b 1'],
        next_embed: Float['b 1 nd'],
        temperature: float = 1.,
        filter_logits_fn: Callable = top_k_2,
        filter_kwargs: dict = dict()
    ):

        def sample_func(logits):

            if logits.ndim == 4:
                enable_squeeze = True
                logits = logits.squeeze(1)
            else:
                enable_squeeze = False

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                sample = filtered_logits.argmax(dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)

                sample = torch.zeros((probs.shape[0], probs.shape[1]), dtype=torch.long, device=probs.device)
                for b_i in range(probs.shape[0]):
                    sample[b_i] = torch.multinomial(probs[b_i], 1).squeeze()

            if enable_squeeze:
                sample = sample.unsqueeze(1)

            return sample

        next_translation_logits = rearrange(self.to_translation_logits(next_embed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_translation)
        next_discretize_translation = sample_func(next_translation_logits)
        next_translation = torch.stack([
            self.undiscretize_translation_x(next_discretize_translation[:, :, 0]),
            self.undiscretize_translation_y(next_discretize_translation[:, :, 1]),
            self.undiscretize_translation_z(next_discretize_translation[:, :, 2])
        ], dim=-1)
        # fill mos positions with nan
        next_translation = next_translation.masked_fill(next_mos.unsqueeze(-1), self.split_id)
        translation_new, _ = pack([translation, next_translation], 'b * nd')

        next_translation_embed = self.translation_embed(next_discretize_translation)
        next_embed_packed, _ = pack([next_embed, next_translation_embed], 'b np *')
        next_width_logits = rearrange(self.to_width_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_width)
        next_discretize_width = sample_func(next_width_logits)
        next_width = self.undiscretize_width(next_discretize_width)
        # fill mos positions with nan
        next_width = next_width.masked_fill(next_mos.unsqueeze(-1), self.split_id)
        width_new, _ = pack([width, next_width], 'b *')

        next_width_embed = self.width_embed(next_discretize_width)
        next_embed_packed, _ = pack([next_embed_packed, next_width_embed], 'b np *')
        next_thickness_logits = rearrange(self.to_thickness_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_thickness)
        next_discretize_thickness = sample_func(next_thickness_logits)
        next_thickness = self.undiscretize_thickness(next_discretize_thickness)
        # fill mos positions with nan
        next_thickness = next_thickness.masked_fill(next_mos.unsqueeze(-1), self.split_id)
        thickness_new, _ = pack([thickness, next_thickness], 'b *')

        return (
            translation_new,
            width_new,
            thickness_new
        )


    @typecheck
    def sample_hair_w_beam_translation(
        self,
        translation: Float['b np 3 nd'],
        width: Float['b np nd'],
        thickness: Float['b np nd'],
        next_mos: Float['b 1'],
        next_embed: Float['b 1 nd'],
        temperature: float = 1.,
        filter_logits_fn: Callable = top_k_2,
        filter_kwargs: dict = dict(),
        beam_size: int = 5,
        return_beam_size: int = 1,
    ):
        b = translation.shape[0]
        assert b == 1, "batch_size must be 1"
        
        log_probs = torch.zeros(beam_size, device=translation.device)

        next_translation_logits = rearrange(self.to_translation_logits(next_embed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_translation)
        log_probs_xyz = F.log_softmax(next_translation_logits / temperature, dim=-1)  # (1, 1, 3, nd)
        topk_log_probs, topk_indices = torch.topk(log_probs_xyz.squeeze(1).squeeze(0), beam_size)  # (3, beam), (3, beam)
        topk_log_probs_x, topk_log_probs_y, topk_log_probs_z = topk_log_probs
        topk_log_probs_x = topk_log_probs_x[:, None, None]
        topk_log_probs_y = topk_log_probs_y[None, :, None]
        topk_log_probs_z = topk_log_probs_z[None, None, :]
        topk3_log_probs = (topk_log_probs_x + topk_log_probs_y + topk_log_probs_z).view(-1)
        log_probs, topk3_topk_indices = torch.topk(topk3_log_probs, beam_size)
        topk3_topk_indices_x = topk3_topk_indices // (beam_size * beam_size)
        topk3_topk_indices_y = (topk3_topk_indices % (beam_size * beam_size)) // beam_size
        topk3_topk_indices_z = topk3_topk_indices % beam_size
        topk_indices_x = topk_indices[0][topk3_topk_indices_x]
        topk_indices_y = topk_indices[1][topk3_topk_indices_y]
        topk_indices_z = topk_indices[2][topk3_topk_indices_z]
        next_discretize_translation = torch.stack([topk_indices_x, topk_indices_y, topk_indices_z], dim=1).unsqueeze(1)

        next_embed = next_embed.repeat(beam_size, 1, 1)

        next_trans_embed = self.translation_embed(next_discretize_translation)
        next_embed_packed, _ = pack([next_embed, next_trans_embed], 'b np *')
        next_width_logits = rearrange(self.to_width_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_width)
        next_discretize_width = next_width_logits.argmax(dim=-1)

        next_width_embed = self.width_embed(next_discretize_width)
        next_embed_packed, _ = pack([next_embed_packed, next_width_embed], 'b np *')
        next_thickness_logits = rearrange(self.to_thickness_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_thickness)
        next_discretize_thickness = next_thickness_logits.argmax(dim=-1)

        next_translation = torch.stack([self.undiscretize_translation_x(next_discretize_translation[:, :, 0]),
                                        self.undiscretize_translation_y(next_discretize_translation[:, :, 1]),
                                        self.undiscretize_translation_z(next_discretize_translation[:, :, 2])], dim=-1).masked_fill(next_mos.unsqueeze(-1), self.split_id)
        
        next_width = self.undiscretize_width(next_discretize_width).masked_fill(next_mos.unsqueeze(-1), self.split_id)
        next_thickness = self.undiscretize_thickness(next_discretize_thickness).masked_fill(next_mos.unsqueeze(-1), self.split_id)

        return (
            next_translation,  # [K, 1, 3]
            next_width,  # [K, 1, 1]
            next_thickness,  # [K, 1, 1]
            log_probs  # [K]
        )
    

    @eval_decorator
    @torch.no_grad()
    @typecheck
    def generate(
        self,
        batch_size: int | None = None,
        filter_logits_fn: Callable = top_k_2,
        filter_kwargs: dict = dict(),
        temperature: float = 1.,
        translation: Float['b np 3'] | None = None,
        width: Float['b np'] | None = None,
        thickness: Float['b np'] | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        cache_kv = True,
        max_seq_len = None,
        strategies: dict = {}
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(translation) and exists(width) and exists(thickness):
            assert not exists(batch_size)
            assert translation.shape[1] == width.shape[1] == thickness.shape[1]
            assert translation.shape[1] <= self.max_seq_len
            
            batch_size = translation.shape[0]

        if self.condition_on_shape:
            assert exists(pc) ^ exists(pc_embed), '`pc` or `pc_embed` must be passed in'
            if exists(pc):
                pc_embed = self.embed_pc(pc)

            batch_size = default(batch_size, pc_embed.shape[0])

        batch_size = default(batch_size, 1)

        translation = default(translation, torch.empty((batch_size, 0, 3), dtype=torch.float32, device=self.device))
        width = default(width, torch.empty((batch_size, 0), dtype=torch.float32, device=self.device))
        thickness = default(thickness, torch.empty((batch_size, 0), dtype=torch.float32, device=self.device))
        mos_codes = torch.zeros((batch_size, 0), dtype=torch.bool, device=self.device)

        curr_length = translation.shape[1]

        cache = None
        eos_codes = None
        mos_cnt = 0

        for i in tqdm(range(curr_length, max_seq_len)):
            can_eos = i != 0 and mos_cnt >= 10

            output = self.forward(
                translation=translation,
                width=width,
                thickness=thickness,
                pc_embed=pc_embed,
                return_loss=False,
                return_cache=cache_kv,
                append_eos=False,
                cache=cache
            )

            if cache_kv:
                next_embed, cache = output
            else:
                next_embed = output

            next_mos_logits = self.to_mos_logits(next_embed).squeeze(-1)
            next_mos_code = (F.sigmoid(next_mos_logits) > 0.5)
            mos_codes = safe_cat([mos_codes, next_mos_code], 1)

            if i != 0 and mos_codes.squeeze()[-1] == True:
                print('MOS!', i)
                mos_cnt += 1

            (
                translation,
                width,
                thickness
            ) = self.sample_hair(
                translation,
                width,
                thickness,
                next_mos_code,
                next_embed,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs
            )

            next_eos_logits = self.to_eos_logits(next_embed).squeeze(-1)
            next_eos_code = (F.sigmoid(next_eos_logits) > 0.5) if mos_cnt >= 10 else torch.zeros_like(next_eos_logits).to(torch.bool)
            eos_codes = safe_cat([eos_codes, next_eos_code], 1)

            # check continuity
            last_mos_idx = torch.nonzero(translation[0, :, 0] < -0.5005)
            if len(last_mos_idx) == 0:
                last_mos_idx = 0
            else:
                last_mos_idx = last_mos_idx.squeeze(-1)[-1] + 1

            if ("long_hair_reparam" in strategies and len(translation[0, last_mos_idx:]) > strategies["long_hair_reparam"]["max_point_num"] and next_mos_code.squeeze().item() == True) \
                or ("long_hair_reparam" in strategies and len(translation[0, last_mos_idx:]) > strategies["long_hair_reparam"]["force_stop_num"]):
                print('long hair, reparam', len(translation[0, last_mos_idx:]))
                force_stop = len(translation[0, last_mos_idx:]) > strategies["long_hair_reparam"]["force_stop_num"]
                cur_hair_translation = translation[0, last_mos_idx:, :]
                cur_hair_width = width[0, last_mos_idx:]
                cur_hair_thickness = thickness[0, last_mos_idx:]
                cur_hair_mos = mos_codes[0, last_mos_idx:]
                cur_hair_eos = eos_codes[0, last_mos_idx:]
                
                # Create evenly spaced points from 0 to 1 for original sequence
                x_old = np.linspace(0, 1, len(cur_hair_translation))
                # Create evenly spaced points for target sequence length
                x_new = np.linspace(0, 1, strategies["long_hair_reparam"]["max_point_num"])
                
                # Interpolate translation coordinates
                new_translation = torch.zeros((strategies["long_hair_reparam"]["max_point_num"], 3), dtype=torch.float32, device=self.device)
                for axis_i in range(3):
                    cs = CubicSpline(x_old, cur_hair_translation[:, axis_i].cpu().numpy())
                    new_translation[:, axis_i] = torch.tensor(cs(x_new), dtype=torch.float32, device=self.device)
                cur_hair_translation = new_translation

                # Interpolate width
                cs_width = CubicSpline(x_old, cur_hair_width.cpu().numpy())
                cur_hair_width = torch.tensor(cs_width(x_new), dtype=torch.float32, device=self.device)
                cur_hair_width = torch.clamp(cur_hair_width, min=0.0)
                
                # Interpolate thickness 
                cs_thickness = CubicSpline(x_old, cur_hair_thickness.cpu().numpy())
                cur_hair_thickness = torch.tensor(cs_thickness(x_new), dtype=torch.float32, device=self.device)
                cur_hair_thickness = torch.clamp(cur_hair_thickness, min=0.0)
                
                # Pad or truncate mos and eos codes to match new length
                new_mos = torch.zeros(strategies["long_hair_reparam"]["max_point_num"], dtype=torch.bool, device=self.device)
                new_eos = torch.zeros(strategies["long_hair_reparam"]["max_point_num"], dtype=torch.bool, device=self.device)
                # Keep first and last values of mos/eos
                if len(cur_hair_mos) > 0:
                    new_mos[0] = cur_hair_mos[0]
                    new_mos[-1] = cur_hair_mos[-1]
                if len(cur_hair_eos) > 0:
                    new_eos[0] = cur_hair_eos[0] 
                    new_eos[-1] = cur_hair_eos[-1]

                # Update all tensors with new interpolated values and truncate to new length
                new_length = last_mos_idx + strategies["long_hair_reparam"]["max_point_num"]
                translation = translation[:, :new_length]
                width = width[:, :new_length] 
                thickness = thickness[:, :new_length]
                mos_codes = mos_codes[:, :new_length]
                eos_codes = eos_codes[:, :new_length]

                translation[0, last_mos_idx:, :] = cur_hair_translation
                width[0, last_mos_idx:] = cur_hair_width
                thickness[0, last_mos_idx:] = cur_hair_thickness
                mos_codes[0, last_mos_idx:] = new_mos
                eos_codes[0, last_mos_idx:] = new_eos

                if force_stop:
                    translation[0, -1, :] = self.split_id
                    width[0, -1] = self.split_id
                    thickness[0, -1] = self.split_id
                    mos_codes[0, -1] = True
                    next_mos_code.fill_(True)
                    mos_cnt += 1
                    print('force mos')

            if can_eos and eos_codes.any(dim=-1).all():
                break

            
            if len(translation[0, last_mos_idx:]) > 5:
                x = np.linspace(0, 1, len(translation[0, last_mos_idx:]))
                extrapolate_point = []
                for axis_i in range(3):
                    cs = CubicSpline(x, translation[0, last_mos_idx:, axis_i].cpu().numpy())
                    extrapolate_point.append(cs(1+1/len(translation[0, last_mos_idx:])).item())

                extrapolate_point = torch.tensor(extrapolate_point, dtype=torch.float32, device=self.device)
                predicted_point = translation[0, -1, :]
                if torch.norm(predicted_point - extrapolate_point) > 0.03:
                    print('not continuous, mos')
                    translation[0, -1, :] = self.split_id
                    width[0, -1] = self.split_id
                    thickness[0, -1] = self.split_id
                    mos_codes[0, -1] = True
                    mos_cnt += 1

            if len(translation[0, last_mos_idx:]) <= 5 and len(translation[0, last_mos_idx:]) != 0 and "root_pc_check" in strategies:
                dist_thres = strategies["root_pc_check"]["dist_thres"]
                beam_size = strategies["root_pc_check"]["beam_size"]
                # check if the point's distance to the pc is too far
                # Get the last point's position
                last_point = translation[0, -1, :]
                
                # Calculate distances to all points in point cloud
                distances = torch.norm(pc[0, :, :3]/2 - last_point.unsqueeze(0), dim=-1)
                
                # Get minimum distance
                min_dist = torch.min(distances)
                
                if min_dist > dist_thres:
                    new_translation, new_width, new_thickness, child_log_probs = self.sample_hair_w_beam_translation(
                        translation=translation[:, :-1],
                        width=width[:, :-1],
                        thickness=thickness[:, :-1],
                        next_mos=next_mos_code,
                        next_embed=next_embed,
                        temperature=strategies["root_pc_check"]["temperature"],
                        filter_logits_fn=filter_logits_fn,
                        filter_kwargs=filter_kwargs,
                        beam_size=beam_size,
                        return_beam_size=beam_size
                    )
                    cur_min, cur_idx = 114514, -1
                    for can_idx in range(beam_size):
                        new_distances = torch.norm(pc[0, :, :3]/2 - new_translation[can_idx, -1, :].unsqueeze(0), dim=-1)
                        min_dist = torch.min(new_distances)
                        if min_dist < cur_min:
                            cur_min = min_dist
                            cur_idx = can_idx
                        if min_dist < dist_thres:
                            break
                        
                    print('replace the last point', min_dist)
                    if cur_idx != -1:
                        translation[0, -1, :] = new_translation[cur_idx, -1, :]
                        width[0, -1] = new_width[cur_idx, -1]
                        thickness[0, -1] = new_thickness[cur_idx, -1]

        # mask out to padding anything after the first eos
        mask = eos_codes.float().cumsum(dim=-1) >= 1
        # concat cur_length to mask
        mask = torch.cat((torch.zeros((batch_size, curr_length), dtype=torch.bool, device=self.device), mask), dim=-1)
        translation = translation.masked_fill(mask.unsqueeze(-1), self.pad_id)
        width = width.masked_fill(mask, self.pad_id)
        thickness = thickness.masked_fill(mask, self.pad_id)

        recon_hair = {
            'translation': translation,
            'width': width,
            'thickness': thickness,
            'mos_codes': mos_codes
        }
        not_eos_mask = ~eos_codes

        return recon_hair, not_eos_mask


    @typecheck
    def encode(
        self,
        *,
        translation: Float['b np 3'],
        width: Float['b np'],
        thickness: Float['b np'],
        not_eos_mask: Bool['b np'],
        return_hair = False
    ):
        """
        einops:
        b - batch
        np - number of hair control points + mos
        c - coordinates (3)
        d - embed dim
        """

        discretize_translation_x = self.discretize_translation_x(translation[..., 0])
        discretize_translation_y = self.discretize_translation_y(translation[..., 1])
        discretize_translation_z = self.discretize_translation_z(translation[..., 2])
        discretize_translation = torch.stack([discretize_translation_x, discretize_translation_y, discretize_translation_z], dim=-1)
        translation_embed = self.translation_embed(discretize_translation)
        translation_embed = rearrange(translation_embed, 'b np c d -> b np (c d)')

        discretize_width = self.discretize_width(width)
        width_embed = self.width_embed(discretize_width)

        discretize_thickness = self.discretize_thickness(thickness)
        thickness_embed = self.thickness_embed(discretize_thickness)

        if self.embed_order == 'xyzwt':
            hair_embed, _ = pack([translation_embed, width_embed, thickness_embed], 'b np *')
        else:
            raise ValueError(f'invalid embed order {self.embed_order}')
        hair_embed = self.project_in(hair_embed)

        hair_embed = hair_embed.masked_fill(~not_eos_mask.unsqueeze(-1), 0.)

        if not return_hair:
            return hair_embed

        hair_embed_unpacked = {
            'translation': translation_embed,
            'width': width_embed,
            'thickness': thickness_embed
        }

        hair_gt = {
            'translation': discretize_translation,
            'width': discretize_width,
            'thickness': discretize_thickness
        }

        return hair_embed, hair_embed_unpacked, hair_gt


    def compute_loss(
        self,
        hair_logits,
        hairs_gt,
        not_mos_mask,
        not_eos_mask,
        pc,
        reduction='mean'
    ):
        
        assert hair_logits['translation'].shape[1] == hairs_gt['translation'].shape[1] + 1 == not_mos_mask.shape[1] + 1 == not_eos_mask.shape[1] + 1

        hairs_gt = {key: pad_tensor(labels) for key, labels in hairs_gt.items()}
        not_eos_mask = pad_tensor(not_eos_mask)
        not_mos_mask = pad_tensor(not_mos_mask)
        hair_mask = not_eos_mask & not_mos_mask

        translation_logits = rearrange(hair_logits['translation'], 'b np c nd -> b nd (np c)')
        width_logits = rearrange(hair_logits['width'], 'b np c nd -> b nd (np c)')
        thickness_logits = rearrange(hair_logits['thickness'], 'b np c nd -> b nd (np c)')
        mos_logits = hair_logits['mos']
        eos_logits = hair_logits['eos']
        translation_gt = rearrange(hairs_gt['translation'], 'b np c -> b 1 (np c)')
        width_gt = rearrange(hairs_gt['width'], 'b np -> b 1 np')
        thickness_gt = rearrange(hairs_gt['thickness'], 'b np -> b 1 np')

        with autocast(enabled = False):

            translation_log_prob = translation_logits.log_softmax(dim=1)
            translation_one_hot = torch.zeros_like(translation_log_prob).scatter(1, translation_gt, 1.)
            width_log_prob = width_logits.log_softmax(dim=1)
            width_one_hot = torch.zeros_like(width_log_prob).scatter(1, width_gt, 1.)
            thickness_log_prob = thickness_logits.log_softmax(dim=1)
            thickness_one_hot = torch.zeros_like(thickness_log_prob).scatter(1, thickness_gt, 1.)

            eos_loss = F.binary_cross_entropy_with_logits(eos_logits, (~not_eos_mask).float(), reduction='none', pos_weight=torch.tensor(self.loss_weight['eos_ce_weight'], device=eos_logits.device))
            eos_mask = not_eos_mask.clone()
            code_lens = not_eos_mask.sum(dim=-1)
            batch_arange = torch.arange(eos_mask.shape[0], device=eos_mask.device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')
            eos_mask[batch_arange, code_lens] = True
            eos_loss = eos_loss * eos_mask

            mos_loss = F.binary_cross_entropy_with_logits(mos_logits, (~not_mos_mask).float(), reduction='none', pos_weight=torch.tensor(self.loss_weight['mos_ce_weight'], device=mos_logits.device))
            mos_loss = mos_loss * not_eos_mask

            if reduction != 'none':
                translation_logits_new = rearrange(hair_logits['translation'], 'b np c nd -> b nd np c')
                translation_gt_new = rearrange(hairs_gt['translation'], 'b np c -> b 1 np c')

                translation_prob = translation_logits_new.softmax(dim=1)
                width_prob = width_logits.softmax(dim=1)
                thickness_prob = thickness_logits.softmax(dim=1)

                translation_indices_x = torch.arange(translation_prob.shape[1], device=translation_prob.device)
                translation_indices_y = torch.arange(translation_prob.shape[1], device=translation_prob.device)
                translation_indices_z = torch.arange(translation_prob.shape[1], device=translation_prob.device)
                width_indices = torch.arange(width_prob.shape[1], device=width_prob.device)
                thickness_indices = torch.arange(thickness_prob.shape[1], device=thickness_prob.device)

                width_loss = (-width_one_hot * width_log_prob).sum(dim=1)[hair_mask]
                thickness_loss = (-thickness_one_hot * thickness_log_prob).sum(dim=1)[hair_mask]
                hair_mask = repeat(hair_mask, 'b np -> b (np c)', c=3)
                translation_loss = (-translation_one_hot * translation_log_prob).sum(dim=1)[hair_mask]

            else:
                width_loss = (-width_one_hot * width_log_prob).sum(dim=1) * hair_mask
                thickness_loss = (-thickness_one_hot * thickness_log_prob).sum(dim=1) * hair_mask
                hair_mask = repeat(hair_mask, 'b np -> b (np c)', c=3)
                translation_loss = (-translation_one_hot * translation_log_prob).sum(dim=1) * hair_mask

        if reduction != 'none':
            eos_loss = eos_loss.sum() / eos_mask.sum()
            mos_loss = mos_loss.sum() / not_eos_mask.sum()
            width_loss = width_loss.mean()
            thickness_loss = thickness_loss.mean()
            translation_loss = translation_loss.mean()
        
        else:
            eos_loss = eos_loss.sum(axis=1) / eos_mask.sum(axis=1)
            mos_loss = mos_loss.sum(axis=1) / not_eos_mask.sum(axis=1)
            width_loss = width_loss.sum(axis=1) / (hair_mask.sum(axis=1) / 3)
            thickness_loss = thickness_loss.sum(axis=1) / (hair_mask.sum(axis=1) / 3)
            translation_loss = translation_loss.sum(axis=1) / hair_mask.sum(axis=1)

        total_loss = self.loss_weight['eos'] * eos_loss \
            + self.loss_weight['mos'] * mos_loss \
            + self.loss_weight['width'] * width_loss \
            + self.loss_weight['thickness'] * thickness_loss \
            + self.loss_weight['translation'] * translation_loss
        loss_breakdown = (eos_loss, mos_loss, width_loss, thickness_loss, translation_loss)
        
        return total_loss, loss_breakdown


    def forward(
        self,
        *,
        translation: Float['b np 3'],
        width: Float['b np'],
        thickness: Float['b np'],
        return_loss = True,
        loss_reduction: str = 'mean',
        return_cache = False,
        append_eos = True,
        cache: LayerIntermediates | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        **kwargs
    ):
        
        not_mos_mask = reduce(translation != self.split_id, 'b np 3 -> b np', 'all')
        not_eos_mask = reduce(translation != self.pad_id, 'b np 3 -> b np', 'all')
        mos_codes = torch.logical_not(not_mos_mask)

        if translation.shape[1] > 0:
            codes, hair_embeds, hairs_gt = self.encode(
                translation=translation,
                width=width,
                thickness=thickness,
                not_eos_mask=not_eos_mask,
                return_hair=True
            )
        else:
            codes = torch.empty((translation.shape[0], 0, self.dim), dtype=torch.float32, device=self.device)

        # handle shape conditions

        attn_context_kwargs = dict()

        if self.condition_on_shape:
            assert exists(pc) ^ exists(pc_embed), '`pc` or `pc_embed` must be passed in'

            if exists(pc):
                pc_head, pc_embed = self.conditioner(shape=pc)
                pc_embed = torch.cat([self.to_cond_dim_head(pc_head), self.to_cond_dim(pc_embed)], dim=-2)

            assert pc_embed.shape[0] == codes.shape[0], 'batch size of point cloud is not equal to the batch size of the codes'

            pooled_pc_embed = pc_embed.mean(dim=1) # (b, shape_condition_dim)

            if self.shape_cond_with_cross_attn:
                attn_context_kwargs = dict(
                    context=pc_embed
                )

            if self.coarse_adaptive_rmsnorm:
                attn_context_kwargs.update(
                    condition=pooled_pc_embed
                )

        batch, seq_len, _ = codes.shape # (b, np, dim)
        device = codes.device
        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # replace mos positions with mos token
        b_index, np_index = torch.where(mos_codes)
        codes[b_index, np_index] = self.mos_token

        # auto append eos token
        if append_eos:
            assert exists(codes)

            code_lens = not_eos_mask.sum(dim=-1)
            codes = pad_tensor(codes)

            batch_arange = torch.arange(batch, device=device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')
            codes[batch_arange, code_lens] = self.eos_token # (b, np+1, dim)

        # if returning loss, save the labels for cross entropy
        if return_loss:
            assert seq_len > 0
            codes = codes[:, :-1]

        hair_codes = codes # (b, np, dim)

        hair_codes_len = hair_codes.shape[-2]

        # cache logic
        (
            coarse_cache,
            coarse_gateloop_cache,
            coarse_post_gateloop_cache,
        ) = cache if exists(cache) else ((None,) * 3)

        if not exists(cache):
            sos = repeat(self.sos_token, 'n d -> b n d', b=batch)

            if self.shape_cond_with_cat:
                sos, _ = pack([pc_embed, sos], 'b * d')
            hair_codes, packed_sos_shape = pack([sos, hair_codes], 'b * d') # (b, n_sos+np, dim)

        if self.condition_on_shape:
            hair_codes = self.shape_coarse_film_cond(hair_codes, pooled_pc_embed)


        if exists(self.coarse_gateloop_block):
            hair_codes, coarse_gateloop_cache = self.coarse_gateloop_block(hair_codes, cache=coarse_gateloop_cache)

        attended_hair_codes, coarse_cache = self.decoder( # (b, n_sos+np, dim) 
            hair_codes,
            cache=coarse_cache,
            return_hiddens=True,
            **attn_context_kwargs
        )

        if exists(self.coarse_post_gateloop_block):
            hair_codes, coarse_post_gateloop_cache = self.coarse_post_gateloop_block(hair_codes, cache=coarse_post_gateloop_cache)

        embed = attended_hair_codes[:, -(hair_codes_len + 1):] # (b, np+1, dim)

        if return_loss:
            translation_embed = hair_embeds['translation']
            width_embed = hair_embeds['width']
            thickness_embed = hair_embeds['thickness']
            if append_eos:
                translation_embed = pad_tensor(translation_embed)
                width_embed = pad_tensor(width_embed)
                thickness_embed = pad_tensor(thickness_embed)

            eos_logits = self.to_eos_logits(embed).squeeze(-1)
            mos_logits = self.to_mos_logits(embed).squeeze(-1)

            translation_logits = rearrange(self.to_translation_logits(embed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_translation)

            embed_packed, _ = pack([embed, translation_embed], 'b np *')
            width_logits = rearrange(self.to_width_logits(embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_width)
            embed_packed, _ = pack([embed_packed, width_embed], 'b np *')
            thickness_logits = rearrange(self.to_thickness_logits(embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_thickness)

            logits = {
                'eos': eos_logits,
                'mos': mos_logits,
                'translation': translation_logits,
                'width': width_logits,
                'thickness': thickness_logits
            }

        if not return_loss:
            if not return_cache:
                return embed[:, -1:]

            next_cache = (
                coarse_cache,
                coarse_gateloop_cache,
                coarse_post_gateloop_cache
            )
            return embed[:, -1:], next_cache

        return self.compute_loss(
            logits,
            hairs_gt,
            not_mos_mask,
            not_eos_mask,
            pc,
            reduction=loss_reduction
        )


def pad_tensor(tensor):
    if tensor.dim() == 3:
        bs, seq_len, dim = tensor.shape
        padding = torch.zeros((bs, 1, dim), dtype=tensor.dtype, device=tensor.device)
    elif tensor.dim() == 2:
        bs, seq_len = tensor.shape
        padding = torch.zeros((bs, 1), dtype=tensor.dtype, device=tensor.device)
    else:
        raise ValueError('Unsupported tensor shape: {}'.format(tensor.shape))
    
    return torch.cat([tensor, padding], dim=1)
