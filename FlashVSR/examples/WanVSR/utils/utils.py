from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import math

def safe_cpu_interpolate(tensor: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    低显存插值：把张量搬到 CPU 用 float32 做插值，再搬回原 device，
    并在需要时降精度为 bf16/原 dtype。
    """
    dev = tensor.device
    orig_dtype = tensor.dtype
    # 插值在 CPU 上用 FP32，避免 XPU 大中间张量
    t_cpu = tensor.to('cpu', dtype=torch.float32, non_blocking=False)
    out_cpu = F.interpolate(t_cpu, size=out_hw, mode='nearest')  # 与原代码一致：nearest
    # 回到原 device，优先 bf16（更省显存）
    if orig_dtype in (torch.float16, torch.bfloat16):
        return out_cpu.to(dev, dtype=torch.bfloat16, non_blocking=False)
    else:
        return out_cpu.to(dev, dtype=orig_dtype, non_blocking=False)
    
# ---------- 新增：分片插值，进一步压低峰值 ----------
def chunked_interpolate(tensor: torch.Tensor, out_hw: tuple[int, int], tile_w: int = 512) -> torch.Tensor:
    """
    在 CPU/FP32 上沿宽度方向分片做最近邻插值，避免一次性巨幅放大导致 OOM。
    期望张量形状 [B, H, Hq, Hk]（最后两维是需要插值的 2D）。
    """
    dev = tensor.device
    orig_dtype = tensor.dtype
    tgt_h, tgt_w = out_hw

    t_cpu = tensor.to('cpu', dtype=torch.float32, non_blocking=False)
    b, heads, h, w = t_cpu.shape
    out_cpu = torch.empty((b, heads, tgt_h, tgt_w), dtype=torch.float32, device='cpu')

    for start in range(0, tgt_w, tile_w):
        end = min(start + tile_w, tgt_w)
        # 近似反推源宽度切片（nearest 足够）
        src_start = int(start * w / tgt_w)
        src_end = max(src_start + 1, int(end * w / tgt_w))
        src_slice = t_cpu[:, :, :, src_start:src_end]
        up_slice = F.interpolate(src_slice, size=(tgt_h, end - start), mode='nearest')
        out_cpu[:, :, :, start:end] = up_slice

    if orig_dtype in (torch.float16, torch.bfloat16):
        return out_cpu.to(dev, dtype=torch.bfloat16, non_blocking=False)
    else:
        return out_cpu.to(dev, dtype=orig_dtype, non_blocking=False)
    
def _sdpa_striped(q, k, v, attn_mask, dropout_p, is_causal, stripe_q=128):
    """
    q,k,v: [B,H,Sq,D] / [B,H,Sk,D]
    attn_mask: None 或 [B,H,mask_h,mask_w]，mask_h/w 可以与 Sq/Sk 不同
    逐条带计算 SDPA，降低显存峰值
    """
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    out_chunks = []
    # 预取 CPU 侧的掩码以避免重复 .cpu()
    mask_cpu = None
    if attn_mask is not None:
        mask_cpu = attn_mask.detach().to('cpu', dtype=torch.float32, non_blocking=False)

    for qs in range(0, Sq, stripe_q):
        qe = min(qs + stripe_q, Sq)
        q_s = q[:, :, qs:qe, :]  # [B,H,qs,D]

        attn_mask_s = None
        if mask_cpu is not None:
            # 仅把本条带的掩码高度插值到 (qe-qs)，宽度插到 Sk，都在 CPU/FP32 做
            mh, mw = mask_cpu.shape[-2:]
            if (mh != (qe - qs)) or (mw != Sk):
                m_slice_cpu = F.interpolate(
                    mask_cpu, size=(qe - qs, Sk), mode='nearest'
                )
            else:
                # 高度正好匹配整段，则直接切条带（快很多）
                m_slice_cpu = mask_cpu[..., qs:qe, :]

            # 搬回 XPU，并对齐 dtype
            attn_mask_s = m_slice_cpu.to(q_s.device, dtype=q_s.dtype, non_blocking=False)

        # 核心：每次只计算一条带，减小 B*H*Sq*Sk 的中间张量
        out_s = F.scaled_dot_product_attention(
            q_s, k, v, attn_mask=attn_mask_s, dropout_p=0.0, is_causal=is_causal
        )
        out_chunks.append(out_s)

        # 主动释放条带中间量，压低峰值
        del q_s, attn_mask_s, out_s
        if hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        torch.xpu.empty_cache()

    # 拼回完整输出
    return torch.cat(out_chunks, dim=2)  # [B,H,Sq,D]


CACHE_T = 2

def block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                          streaming_info, base_blockmask, max_seqlen_q_, max_seqlen_k_,
                          p_dropout, deterministic=False, softmax_scale=None,
                          is_causal=False, exact_streaming=False, return_attn_probs=False,
                          stripe_q=64):
    """
    以条带(Stripe)方式做 SDPA，显著降低显存峰值。
    假设 batch=1（与原实现一致），输入 q/k/v 原形如 [B*Sq, H, D] / [B*Sk, H, D]。
    """
    B = 1
    Sq = q.shape[0] // B
    Sk = k.shape[0] // B
    H = q.shape[1]
    D = q.shape[2]

    # 统一到 [B,H,S,D]
    q = q.view(B, Sq, H, D).transpose(1, 2).contiguous()  # [B,H,Sq,D]
    k = k.view(B, Sk, H, D).transpose(1, 2).contiguous()  # [B,H,Sk,D]
    v = v.view(B, Sk, H, D).transpose(1, 2).contiguous()  # [B,H,Sk,D]

    # 掩码常驻 CPU；每个条带再切片+插值+小片搬回 XPU
    mask_cpu = None
    if base_blockmask is not None:
        m = base_blockmask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        elif m.dim() == 3:
            m = m.unsqueeze(0)               # [1,*,h,w]
        mask_cpu = m.detach().to('cpu', dtype=torch.float32, non_blocking=False)

    out_chunks = []
    for qs in range(0, Sq, stripe_q):
        qe = min(qs + stripe_q, Sq)
        q_s = q[:, :, qs:qe, :]  # [B,H,qs,D]
        
        attn_mask_s = None
        if mask_cpu is not None:
            mh, mw = mask_cpu.shape[-2], mask_cpu.shape[-1]     # 原掩码高宽
            # 与掩码尺寸取 min，避免空切片
            qs0 = min(qs, mh)
            qe0 = min(qe, mh)

            # 只有当切片后高度>0 且 原始宽度>0 才处理，否则本条带不使用掩码
            if (qe0 > qs0) and (mw > 0):
                m_slice_cpu = mask_cpu[..., qs0:qe0, :]         # [B,*, h_slice, mw]

                target_h = (qe - qs)
                target_w = Sk

                # 需要的话再插值：前提是输入高宽都 >0
                if (m_slice_cpu.shape[-2] != target_h) or (m_slice_cpu.shape[-1] != target_w):
                    if (m_slice_cpu.shape[-2] > 0) and (m_slice_cpu.shape[-1] > 0):
                        m_slice_cpu = F.interpolate(
                            m_slice_cpu, size=(target_h, target_w), mode='nearest'
                        )
                    else:
                        m_slice_cpu = None

                if m_slice_cpu is not None:
                    attn_mask_s = m_slice_cpu.to(q_s.device, dtype=q_s.dtype, non_blocking=False)
            # else: attn_mask_s 维持 None（该条带不施加掩码）

        # 条带 SDPA（推理 dropout=0）
        out_s = F.scaled_dot_product_attention(
            q_s, k, v,
            attn_mask=attn_mask_s,
            dropout_p=0.0,
            is_causal=is_causal
        )  # [B,H,qs,D]
        out_chunks.append(out_s)

        # 释放条带中间量
        del q_s, attn_mask_s, out_s
        if hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        torch.xpu.empty_cache()

    # 拼回 [B,H,Sq,D] -> [B*Sq,H,D] -> 与原接口一致
    output = torch.cat(out_chunks, dim=2)
    output = output.transpose(1, 2).contiguous().view(B * Sq, H, D)
    return output.squeeze(0)

def chunked_interpolate(tensor, target_size, chunk_size=1024):
    """
    分块插值以减少内存消耗
    """
    batch, heads, h, w = tensor.shape
    target_h, target_w = target_size
    
    output = torch.zeros(batch, heads, target_h, target_w, dtype=tensor.dtype, device=tensor.device)
    
    chunk_h = min(chunk_size, h)
    chunk_w = min(chunk_size, w)
    
    for i in range(0, h, chunk_h):
        for j in range(0, w, chunk_w):
            end_i = min(i + chunk_h, h)
            end_j = min(j + chunk_w, w)

            chunk = tensor[:, :, i:end_i, j:end_j]
            

            target_end_i = int(target_h * end_i / h)
            target_start_i = int(target_h * i / h)
            target_end_j = int(target_w * end_j / w)
            target_start_j = int(target_w * j / w)
            

            if chunk.numel() > 0:
                interpolated = F.interpolate(
                    chunk.float(),
                    size=(target_end_i - target_start_i, target_end_j - target_start_j),
                    mode='nearest'
                ).to(tensor.dtype)
                

                output[:, :, target_start_i:target_end_i, target_start_j:target_end_j] = interpolated
                
    return output


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            # print(cache_x.shape, x.shape)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            # print('cache!')
        x = F.pad(x, padding, mode='replicate') # mode='replicate'
        # print(x[0,0,:,0,0])

        return super().forward(x)
    
class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        # x: (B, C, F, H, W)
        return rearrange(x, 
                         'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
                         ff=self.ff, hh=self.hh, ww=self.ww)

class Buffer_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        # print(out_x.shape)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
    
    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs
