#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import folder_paths
import os, re, time
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange
from .infer_flashvsr_full import tensor_upscale
from ...diffsynth import ModelManager, FlashVSRTinyPipeline
from .utils.utils import Buffer_LQ4x_Proj
from .utils.TCDecoder import build_tcdecoder


####监控工具#####
import csv

VRAM_LOG = os.environ.get("VRAM_LOG", "1") == "1"   # 设置 0 可关闭
VRAM_CSV = os.environ.get("VRAM_CSV", "vram_log.csv")  # 默认写到工作目录

def _bytes_mb(x): 
    return round(x / (1024**2), 1)

def _dev_str(dev):
    return str(dev) if isinstance(dev, str) else f"cuda:{dev.index if hasattr(dev, 'index') else dev}"

class VRAMMeter:
    def __init__(self, devices=("cuda:0","cuda:1")):
        self.devices = [torch.device(d) for d in devices if torch.cuda.device_count() > 0]
        self.rows = []
        self.t0 = time.time()
        # 初始清零峰值
        for d in self.devices:
            with torch.cuda.device(d):
                torch.cuda.reset_peak_memory_stats(d)

    def snap(self, tag):
        if not VRAM_LOG or not torch.cuda.is_available():
            return
        t = time.time() - self.t0
        for d in self.devices:
            alloc = torch.cuda.memory_allocated(d)
            resv  = torch.cuda.memory_reserved(d)
            peak  = torch.cuda.max_memory_allocated(d)
            free,total = torch.cuda.mem_get_info(d)
            self.rows.append({
                "t": round(t,3),
                "tag": tag,
                "device": _dev_str(d),
                "allocated_MB": _bytes_mb(alloc),
                "reserved_MB":  _bytes_mb(resv),
                "peak_MB":      _bytes_mb(peak),
                "free_MB":      _bytes_mb(free),
                "total_MB":     _bytes_mb(total),
            })

    def reset_peak(self, device):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    def to_csv(self, path=VRAM_CSV):
        if not self.rows: 
            return
        fieldnames = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)
        print(f"[VRAMMeter] CSV written -> {os.path.abspath(path)}")

# 全局一个实例（也可放到 run_inference_tiny 里局部创建）
_vram = VRAMMeter(devices=("cuda:0","cuda:1"))





import gc
from contextlib import suppress

def _safe_to(module, device: str):
    if module is None: 
        return
    try:
        module.to(device)
    except Exception:
        # 有些容器/包装器没有 .to，直接忽略
        pass

def _offload_non_decode_modules(pipe, keep=("TCDecoder", "VAE", "ColorCorrector")):
    """
    将不在 keep 列表里的子模块一律下放到 CPU（或直接释放）。
    根据你的 pipe 结构，常见可下放的模块例如：
      - UNet / SRNet / EnhanceNet / Refiners
      - FlowNet / OpticalFlow / SpatioTemporal 模块
      - Text/Image encoder、Embedding、KV 缓存
      - 任何不参与 decode 的辅助网络
    """
    # 你自己清楚 pipeline 里有啥，这里尽量“保守识别、广泛下放”
    for name, sub in list(vars(pipe).items()):
        if name in keep:
            continue
        # 跳过非模块对象
        if not hasattr(sub, "to") and not hasattr(sub, "__class__"):
            continue

        # 优先选择“下放到 CPU”，如果你确认后面不再用，也可以直接 del 掉
        with suppress(Exception):
            _safe_to(sub, "cpu")

    # 如果明确不再需要某些重型模块，可以直接释放引用（更激进）
    maybe_delete = ["UNet", "SRNet", "Enhancer", "FlowNet", "OpticalFlow", 
                    "FeatureExtractor", "Refiner", "TextEncoder", "KVCache",
                    "KernelUp", "PriorNet", "AuxNet"]
    for k in maybe_delete:
        if k in keep:
            continue
        if hasattr(pipe, k):
            try:
                delattr(pipe, k)
            except Exception:
                pass

    # 额外：把可能挂在 pipe 之外的重型缓存也清理（按你的代码环境调整）
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ====== 新增：放在文件顶部其它函数旁，或放到 run_inference_tiny 上方 ======
def _pick_secondary_decode_device(primary_device: str = "cuda:0") -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 1
    # 简单策略：优先选 cuda:1 且不同于 primary
    if n >= 2 and primary_device != "cuda:1":
        return "cuda:1"
    return None

def _is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    # 类型判断（不同 torch 版本类名略有差异，全部兜底）
    oom_type = (
        getattr(torch, "OutOfMemoryError", RuntimeError),
        getattr(torch.cuda, "OutOfMemoryError", RuntimeError),
        RuntimeError,
    )
    if isinstance(e, oom_type):
        # 常见关键词尽量全：包含你的 “Allocation on device”
        keywords = [
            "out of memory",
            "cuda error: out of memory",
            "allocation on device",
            "cublas status alloc failed",
            "c10::error",  # 有些版本会把 OOM 包成 c10::Error
        ]
        return any(k in msg for k in keywords)
    # 再保险：纯字符串匹配
    return any(k in msg for k in [
        "out of memory", "allocation on device", "cuda error: out of memory",
        "cublas status alloc failed", "c10::error"
    ])


# ===== precision config =====
AMP_DTYPE = "bf16"  # 可改为 "fp16"

def _amp_dtype():
    import torch
    return torch.bfloat16 if AMP_DTYPE.lower() == "bf16" else torch.float16

# 轻微的 matmul 优化（Ada 40 系列可用）
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from contextlib import contextmanager

@contextmanager
def _amp_cast(dtype):
    """
    兼容老/新 PyTorch:
    - 新版: torch.autocast("cuda", dtype=...)
    - 旧版: torch.cuda.amp.autocast(dtype=...)
    """
    import torch
    # 新式 API
    if hasattr(torch, "autocast"):
        with torch.autocast("cuda", dtype=dtype):
            yield
    else:
        # 旧式 API
        from torch.cuda.amp import autocast as _old_autocast
        with _old_autocast(dtype=dtype):
            yield


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    try:
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    except:
        try:
            frames=frames.cpu()
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).numpy().astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
            return frames
        except:
            batch_size = min(32, frames.shape[0]) 
            total_frames = frames.shape[0]
            frame_list = []
            for i in range(0, total_frames, batch_size):
                batch_frames = frames[i:min(i + batch_size, total_frames)]
                batch_frames = ((batch_frames.float() + 1) * 127.5).clip(0, 255)
                batch_frames_np = batch_frames.cpu().numpy().astype(np.uint8)
                for frame in batch_frames_np:
                    frame_list.append(Image.fromarray(frame))
            return frame_list

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def dup_first_frame_1cthw_simple(video_tensor):
    return torch.cat([video_tensor[:, :, :1], video_tensor], dim=2)

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype=dtype)

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def prepare_input_tensor(path: str, scale: float = 4,fps=30, dtype=torch.bfloat16, device='cuda'):

    if isinstance(path,torch.Tensor):
        total,h0,w0,_ = path.shape
        if total == 1:
            print("got image,repeating to 25 frames")
            path = path.repeat(25, 1, 1, 1) 
            total=25
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        pil_list=tensor2pillist(path)

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        idx = idx[:F]
        frames = []
        pil_list = [pil_list[i] for i in idx]
        for i in idx:
            img = pil_list[i].convert('RGB')
            img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        frames = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W  
        torch.cuda.empty_cache()
        return frames, tH, tW, F, fps
        
    elif os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
        fps = 30
        return vid, tH, tW, F, fps

    elif is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try: meta = rdr.get_meta_data()
        except Exception: pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf,int) and nf>0: return nf
            except Exception: pass
            try: return r.count_frames()
            except Exception:
                n=0
                try:
                    while True: r.get_data(n); n+=1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total-1]*4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try: rdr.close()
            except Exception: pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, F, fps
    else:
        raise ValueError(f"Unsupported input: {path}")
   

def init_pipeline_tiny(prompt_path,LQ_proj_in_path = "./FlashVSR/LQ_proj_in.ckpt",ckpt_path="./FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors",TCDecoder_path="./FlashVSR/TCDecoder.ckpt",device="cuda"):

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([ckpt_path,])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu",weights_only=False,), strict=True)
    
    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load(TCDecoder_path,weights_only=False,), strict=False)
    print(mis)

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path); pipe.load_models_to_device(["dit","vae"])
    return pipe

def run_inference_tiny(pipe,input,seed,scale,kv_ratio=3.0,local_range=9,step=1,cfg_scale=1.0,sparse_ratio=2.0,color_fix=True,fix_method="wavelet",split_num=81,dtype=torch.bfloat16,device="cuda", save_vodeo_=False,):
    pipe.to('cuda')

    _vram = VRAMMeter(devices=("cuda:0","cuda:1"))
    _vram.snap("start")

    pad_first_frame = True  if "wavelet"== fix_method and color_fix else False

    torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    LQ, th, tw, F, fps = prepare_input_tensor(input, scale=scale,dtype=dtype, device=device)
    frames,LQ_cur_idx = pipe(
        prompt="", negative_prompt="", cfg_scale=cfg_scale, num_inference_steps=step, seed=seed,
        LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio*768*1280/(th*tw), 
        kv_ratio=kv_ratio,
        local_range=local_range,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
        color_fix = color_fix,
    )
    pipe.dit.to('cpu')  
    torch.cuda.empty_cache()   
    #print(LQ.shape,frames.shape,LQ_cur_idx)#torch.Size([1, 3, 81, 384, 640]) torch.Size([1, 16, 20, 48, 80]) 77


    # ====== 替换 run_inference_tiny 内部 with torch.no_grad(): 这一整段 try/except ======
    # ====== decoding with fallback (same logic) + VRAM marks ======
    with torch.no_grad():
        def _decode_on_device(target_device: str, use_segment_fallback: bool = True):
            """在指定 device 上尝试解码；必要时用分段解码回退。返回解码后的 frames (1, C, T, H, W) in [-1,1]"""
            # 把 TCDecoder 和需要参与解码的张量搬到目标设备
            pipe.TCDecoder.to(target_device)
            pipe.TCDecoder.clean_mem()  # ← 新增：换卡后清空 mem

            _vram.reset_peak(target_device)
            _vram.snap(f"decode_enter:{target_device}")

            from torch.cuda.amp import autocast
            amp_dtype = _amp_dtype()

            fr = frames.transpose(1, 2).to(target_device, dtype=amp_dtype, non_blocking=True)          # [B,T,C,H,W] latent
            cond_frames = LQ[:, :, :LQ_cur_idx, :, :].to(target_device, dtype=amp_dtype, non_blocking=True)

            try:
                with _amp_cast(_amp_dtype()):
                    dec = pipe.TCDecoder.decode_video(
                       fr, parallel=False, show_progress_bar=False, cond=cond_frames
                )
                out = dec.transpose(1, 2).mul_(2).sub_(1)  # [B,C,T,H,W] & [-1,1]
                _vram.snap(f"decode_done_once:{target_device}")
                return out
            except RuntimeError as e:
                # 只有 OOM 才考虑分段回退；非 OOM 直接抛出
                is_oom = _is_oom_error(e)
                if not (is_oom and use_segment_fallback):
                    raise

                # 分段回退（在 target_device 上）
                print(f"TCDecoder OOM on {target_device}. Falling back to segmented decoding...")
                segment_size = (split_num - 1) * 2       # e.g. 160
                latent_segment_size = max(1, segment_size // 4)  # e.g. 40
                decoded_chunks_cpu = []
                total_latent_frames = fr.shape[1]

                for start_idx in range(0, total_latent_frames, latent_segment_size):
                    end_idx = min(start_idx + latent_segment_size, total_latent_frames)

                    fr_seg = fr[:, start_idx:end_idx, :, :, :]  # [B, t, C, H, W]

                    start_cond_idx = start_idx * 4
                    end_cond_idx = min(end_idx * 4, LQ_cur_idx)
                    cond_seg = cond_frames[:, :, start_cond_idx:end_cond_idx, :, :]

                    pipe.TCDecoder.clean_mem()  # ← 新增：每段前清空 mem



                    with _amp_cast(_amp_dtype()):
                        dec_seg = pipe.TCDecoder.decode_video(
                            fr_seg, parallel=False, show_progress_bar=False, cond=cond_seg
                        )
                    dec_seg = dec_seg.to("cpu", non_blocking=True)  # 及时搬回 CPU，释放显存
                    decoded_chunks_cpu.append(dec_seg)

                    # 段完成就把这些临时显存清掉
                    del fr_seg, cond_seg, dec_seg
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _vram.snap(f"seg_post_cpu_dump:{target_device}:{start_idx}-{end_idx}")

                dec_all = torch.cat(decoded_chunks_cpu, dim=1)      # CPU 上拼接
                out = dec_all.transpose(1, 2).mul_(2).sub_(1)       # 仍在 CPU
                _vram.snap("cpu_cat_done")

                # 需要回到后续设备（通常是 LQ.device/cuda:0）再继续
                target_after = str(LQ.device)
                out = out.to(target_after, non_blocking=True)

                return out

        try:
            # 1) 先在当前流程设备上尝试一次（通常是 cuda:0）
            # —— 解码前瘦身：只保留 TCDecoder / VAE / ColorCorrector —— 
            _offload_non_decode_modules(pipe, keep=("TCDecoder", "VAE", "ColorCorrector"))
            # 可选：如果 ColorCorrector 用得很晚，也可以先下放，等用到时再搬回
            # _safe_to(pipe.ColorCorrector, "cpu")
            # 需要使用时再：pipe.ColorCorrector.to(LQ.device)
            frames = _decode_on_device(target_device=LQ.device, use_segment_fallback=False)
        except RuntimeError as e0:
            is_oom0 = _is_oom_error(e0)
            if not is_oom0:
                # 非 OOM 直接重抛
                raise

            # 2) OOM 时尝试把解码搬到次卡（cuda:1）
            dev1 = _pick_secondary_decode_device(primary_device=str(LQ.device))
            if dev1 is None:
                print("TCDecoder decode OOM and no secondary GPU available. Trying segmented decode on primary...")
                # 在原卡上改用分段解码
                # —— 解码前瘦身：只保留 TCDecoder / VAE / ColorCorrector —— 
                _offload_non_decode_modules(pipe, keep=("TCDecoder", "VAE", "ColorCorrector"))
                # 可选：如果 ColorCorrector 用得很晚，也可以先下放，等用到时再搬回
                # _safe_to(pipe.ColorCorrector, "cpu")
                # 需要使用时再：pipe.ColorCorrector.to(LQ.device)
                frames = _decode_on_device(target_device=LQ.device, use_segment_fallback=True)
            else:
                print(f"TCDecoder decode OOM on {LQ.device}. Retrying on {dev1}...")
                # —— 解码前瘦身：只保留 TCDecoder / VAE / ColorCorrector —— 
                _offload_non_decode_modules(pipe, keep=("TCDecoder", "VAE", "ColorCorrector"))
                # 可选：如果 ColorCorrector 用得很晚，也可以先下放，等用到时再搬回
                # _safe_to(pipe.ColorCorrector, "cpu")
                # 需要使用时再：pipe.ColorCorrector.to(LQ.device)

                frames = _decode_on_device(target_device=dev1, use_segment_fallback=True)
                # 解码完成，把结果搬回 LQ.device，方便后续 ColorCorrector 使用
                frames = frames.to(LQ.device, non_blocking=True)

    # 颜色校正保持不变（frames 已经被搬回 LQ.device）




    # 颜色校正（wavelet）
    # shape: 1,16, 20, 64, 96
    try:
        if color_fix:
            if pad_first_frame: # 加帧
                frames = dup_first_frame_1cthw_simple(frames)
                LQ=dup_first_frame_1cthw_simple(LQ)
            frames = pipe.ColorCorrector(
                frames.to(device=LQ.device),
                LQ[:, :, :frames.shape[2], :, :],
                clip_range=(-1, 1),
                chunk_size=16,
                method=fix_method
                )
            if pad_first_frame: #减帧
                frames = frames[:, :, 1:, :, :] # remove first frame
    except:
        pass
    print("Done.")
    pipe.TCDecoder.to('cpu')
    del LQ
    torch.cuda.empty_cache()   
    frames = tensor2video(frames[0]) 
    if save_vodeo_:
        save_video(frames, os.path.join(folder_paths.get_output_directory(),f"FlashVSR_Full_seed{seed}.mp4"), fps=fps, quality=6)
    _vram.snap("end")
    _vram.to_csv()
    return frames
