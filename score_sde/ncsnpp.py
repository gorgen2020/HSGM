from . import layers, layerspp, normalization
import torch.nn as nn
from torch.cuda.amp import autocast
import functools
import torch
import numpy as np

from util.utils import get_input_size, get_channel_multiplier, init_temb_fun, mask_inactive_variables
import torch.nn.functional as F
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class CrossAttnBlockpp(nn.Module):
  def __init__(self, channels, cond_channels=1, init_scale=0., skip_rescale=True):
    super().__init__()
    self.channels = channels
    self.skip_rescale = skip_rescale

    self.norm = nn.GroupNorm(min(channels // 4, 32), channels, eps=1e-6)
    self.cond_proj = nn.Conv2d(cond_channels, channels, kernel_size=1)

    self.q = nn.Conv2d(channels, channels, 1)
    self.k = nn.Conv2d(channels, channels, 1)
    self.v = nn.Conv2d(channels, channels, 1)
    self.proj_out = nn.Conv2d(channels, channels, 1)

    # Initialize parameters
    nn.init.zeros_(self.proj_out.weight)
    nn.init.zeros_(self.proj_out.bias)
    if init_scale != 0:
      self.proj_out.weight.data *= init_scale
      self.proj_out.bias.data *= init_scale

  def forward(self, x, cond):
    B, C, H, W = x.shape

    # Adjust condition shape
    cond = F.interpolate(cond, size=(H, W), mode='bilinear', align_corners=False)
    cond = self.cond_proj(cond)

    h = self.norm(x)
    q = self.q(h).view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
    k = self.k(cond).view(B, C, -1)  # [B, C, HW_cond]
    v = self.v(cond).view(B, C, -1).permute(0, 2, 1)  # [B, HW_cond, C]

    # Attention calculation
    attn = torch.bmm(q, k) * (C ** -0.5)
    attn = F.softmax(attn, dim=-1)
    out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)

    out = self.proj_out(out)
    return (x + out) / np.sqrt(2.0) if self.skip_rescale else x + out


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=1024):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, max_len, d_model)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, :x.size(1)]
    return x


class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, args, num_input_channels):
    super().__init__()
    self.act = act = nn.SiLU()
    self.dataset = args.dataset
    self.num_scales = args.num_scales_dae
    self.num_input_channels = num_input_channels

    self.nf = nf = args.num_channels_dae
    ch_mult = get_channel_multiplier(self.dataset, self.num_scales)[:self.num_scales]
    self.num_res_blocks = num_res_blocks = args.num_cell_per_scale_dae
    self.attn_resolutions = attn_resolutions = (32, )
    dropout = args.dropout
    resamp_with_conv = True    # always True in the original codebase
    self.num_resolutions = num_resolutions = self.num_scales
    assert len(ch_mult) == self.num_scales


    self.input_size = get_input_size(self.dataset)[0] // 2 ** (args.num_preprocess_blocks + args.num_latent_scales - 1)
    self.all_resolutions = all_resolutions = [self.input_size // (2 ** i) for i in range(num_resolutions)]

    self.mixed_prediction = args.mixed_prediction  # This enables mixed prediction
    if self.mixed_prediction:
      init = args.mixing_logit_init * torch.ones(size=[1, num_input_channels, 1, 1])
      self.mixing_logit = torch.nn.Parameter(init, requires_grad=True)
      self.is_active = None
    else:
      self.mixing_logit = None
      self.is_active = None

    fir = args.fir
    fir_kernel = [1, 3, 3, 1]
    init_scale = 0.  # always zero in the original codebase
    self.skip_rescale = skip_rescale = True
    self.resblock_type = resblock_type = 'ddpm'
    self.progressive = progressive = args.progressive
    self.progressive_input = progressive_input = args.progressive_input
    combine_method = args.progressive_combine
    combiner = functools.partial(Combine, method=combine_method)

    self.embedding_dim = args.embedding_dim
    self.embedding_dim_mult = 4
    self.temb_fun = init_temb_fun(args.embedding_type, args.embedding_scale, args.embedding_dim)

    self.cond_encoder = nn.Sequential(
      nn.Conv2d(1, 64, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 256, 3, stride=2, padding=1),  # 下采样到16x16
      nn.ReLU(),
      nn.Conv2d(256, self.nf, 3, stride=2, padding=1)  # 下采样到8x8
    )



    modules = []
    modules.append(nn.Linear(self.embedding_dim, self.embedding_dim * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(self.embedding_dim * 4, self.embedding_dim * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    CrossAttnBlock = functools.partial(CrossAttnBlockpp,
                                       init_scale=init_scale,
                                       skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel, keep_size=True)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel, keep_size=True)

    # Downsample = functools.partial(layerspp.Downsample,
    #                                with_conv=True,
    #                                fir=False,
    #                                keep_size=True)  # 新增参


    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=self.embedding_dim * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=self.embedding_dim * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    channels = self.num_input_channels
    if progressive_input != 'none':
      input_pyramid_ch = channels

    modules.append(conv3x3(channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
          modules.append(CrossAttnBlock(channels=in_ch))  # 新增交叉注意
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)


    self.embed_dim = 16
    self.input_proj = nn.Conv2d(1, 36, kernel_size=1)
    self.proj = nn.Linear(36, self.embed_dim)


    self.proj = nn.Linear(36, self.embed_dim)
    self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=48 * 36)

    self.attn = nn.MultiheadAttention(self.embed_dim, 8, batch_first=True)

    self.ln1 = nn.LayerNorm(self.embed_dim)
    self.ff = nn.Sequential(
      nn.Linear(self.embed_dim, self.embed_dim),
      nn.ReLU(),
      nn.Linear(self.embed_dim, self.embed_dim)
    )
    self.ln2 = nn.LayerNorm(self.embed_dim)
    self.output_proj = nn.Linear(self.embed_dim, 32 * 36)


  def forward(self, x, t, y, mask):

    """
    y: (batch, 1, 32, 32)
    mask: (batch, 1, 32, 32)
    """
    y = self.input_proj(y)  # (B, 32, 32, 32)
    y = y.permute(0, 2, 3, 1)  # (B, 32, 32, 32) -> (B, 32, 32, 32)
    B, H, W, C = y.shape
    y = y.view(B, H * W, C)  # (B, 1024, 32)

    mask = mask.view(B, H * W)  # (B, 1024)

    y = self.proj(y)  # (B, 1024, embed_dim=16)

    y_pos = self.pos_encoder(y)

    key_padding_mask = (mask == 0)

    attn_out, _ = self.attn(y_pos, y_pos, y_pos, key_padding_mask=key_padding_mask)
    y_pos = self.ln1(y_pos + attn_out)

    ff_out = self.ff(y_pos)
    y_pos = self.ln2(y_pos + ff_out)

    y_final = y_pos[:, 0, :]  # (B, embed_dim)

    y_final = self.output_proj(y_final)  # (B, 1024)
    y = y_final.view(B, 1, 32, 36)  #  (B, 1, 32, 32)
    #################################################

    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0

    # time embedding
    if t.dim() == 0:
      t = t.expand(1)

    with autocast(False):
      # TODO we do not apply log to t in fourier features
      temb = self.temb_fun(t)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1

    # mask out inactive variables
    if self.mixed_prediction and self.is_active is not None:
      x = mask_inactive_variables(x, self.is_active)

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)]
    m_idx += 1

    y = self.cond_encoder(y)
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
          # 新增交叉注意力
          h = modules[m_idx](h, y)
          m_idx += 1
        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)

    return h
