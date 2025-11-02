#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn.functional as F
import os, re, time
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange
import folder_paths
from ...diffsynth import ModelManager, FlashVSRFullPipeline
from .utils.utils import Buffer_LQ4x_Proj
from comfy.utils import common_upscale
from safetensors.torch import load_file

# === extra imports for AMP/offload/VRAM ===
import os, csv, time, gc
from contextlib import contextmanager, suppress

# =============== VRAM meter (minimal, low-intrusion) ===============
VRAM_LOG = os.environ.get("VRAM_LOG", "1") == "1"   # set 0 to disable
VRAM_CSV = os.environ.get("VRAM_CSV", "vram_log.csv")

def _bytes_mb(x): 
    return round(x / (1024**2), 1)

class VRAMMeter:
    def __init__(self, devices=("cuda:0","cuda:1")):
        self.enabled = VRAM_LOG and torch.cuda.is_available()
        self.devices = [torch.device(d) for d in devices] if self.enabled else []
        self.rows = []
        self.t0 = time.time()
        if self.enabled:
            for d in self.devices:
                with torch.cuda.device(d):
                    torch.cuda.reset_peak_memory_stats(d)

    def snap(self, tag):
        if not self.enabled: 
            return
        t = round(time.time() - self.t0, 3)
        for d in self.devices:
            alloc = torch.cuda.memory_allocated(d)
            resv  = torch.cuda.memory_reserved(d)
            peak  = torch.cuda.max_memory_allocated(d)
            free,total = torch.cuda.mem_get_info(d)
            self.rows.append({
                "t_sec": t, "tag": tag, "device": str(d),
                "allocated_MB": _bytes_mb(alloc),
                "reserved_MB":  _bytes_mb(resv),
                "peak_MB":      _bytes_mb(peak),
                "free_MB":      _bytes_mb(free),
                "total_MB":     _bytes_mb(total),
            })

    def reset_peak(self, device):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(torch.device(device))

    def to_csv(self, path=VRAM_CSV):
        if not (self.enabled and self.rows):
            return
        fieldnames = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)
        print(f"[VRAMMeter] CSV written -> {os.path.abspath(path)}")
# ===================================================================

# ================= AMP config (bf16/fp16, old/new torch) ===========
AMP_DTYPE = "bf16"  # or "fp16"
def _amp_dtype():
    return torch.bfloat16 if AMP_DTYPE.lower() == "bf16" else torch.float16

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
with suppress(Exception):
    torch.set_float32_matmul_precision("high")

@contextmanager
def _amp_cast(dtype):
    # torch>=2.0
    if hasattr(torch, "autocast"):
        with torch.autocast("cuda", dtype=dtype):
            yield
    else:
        # older torch
        from torch.cuda.amp import autocast as _old
        with _old(dtype=dtype):
            yield
# ===================================================================

# ================== Offload & OOM helpers ==========================
def _safe_to(module, device: str):
    if module is None:
        return
    with suppress(Exception):
        module.to(device)

def _offload_non_decode_modules(pipe, keep=("VAE", "ColorCorrector")):
    # Offload anything not needed for decode to CPU
    for name, sub in list(vars(pipe).items()):
        if name in keep:
            continue
        if hasattr(sub, "to") or hasattr(sub, "__class__"):
            _safe_to(sub, "cpu")
    # Aggressive: delete heavy attributes if present
    for k in ["UNet","SRNet","Enhancer","FlowNet","OpticalFlow",
              "FeatureExtractor","Refiner","TextEncoder","KVCache",
              "KernelUp","PriorNet","AuxNet","dit"]:
        if k in keep: 
            continue
        if hasattr(pipe, k):
            with suppress(Exception):
                delattr(pipe, k)
    gc.collect()
    with suppress(Exception):
        torch.cuda.empty_cache()

def _pick_secondary_decode_device(primary_device: str = "cuda:0") -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 1
    if n >= 2 and primary_device != "cuda:1":
        return "cuda:1"
    return None

def _is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    kw = ["out of memory","allocation on device","cuda error: out of memory",
          "cublas status alloc failed","c10::error"]
    return any(k in msg for k in kw)

def _move_vae(pipe, device: str):
    """
    Move VAE to the exact target device.
    Important: WanVAE.to_cuda() always uses 'cuda:0', so DO NOT call it when we need cuda:1.
    """
    vae_attr = "VAE" if getattr(pipe, "new_decoder", False) else "vae"
    vae = getattr(pipe, vae_attr, None)
    if vae is None:
        return

    cls_name = getattr(vae, "__class__", type("X",(object,),{})).__name__

    # diffusers AutoencoderKLWan 或 其它：常规 .to() 就行
    if cls_name != "WanVAE":
        with suppress(Exception):
            vae.to(device)
        return

    # ---- 专门处理 WanVAE：显式搬到目标设备，而不是用 to_cuda()/to_cpu() ----
    target = device
    if device.startswith("cuda"):
        # model & 子模块
        with suppress(Exception):
            vae.model.to(target)
        with suppress(Exception):
            vae.model.encoder.to(target)
        with suppress(Exception):
            vae.model.decoder.to(target)
        # 非注册常量
        with suppress(Exception):
            vae.mean = vae.mean.to(target, non_blocking=True)
            vae.inv_std = vae.inv_std.to(target, non_blocking=True)
            vae.scale = [vae.mean, vae.inv_std]
    else:
        # 回 CPU
        with suppress(Exception):
            vae.model.to("cpu")
        with suppress(Exception):
            vae.model.encoder.to("cpu")
        with suppress(Exception):
            vae.model.decoder.to("cpu")
        with suppress(Exception):
            vae.mean = vae.mean.cpu()
            vae.inv_std = vae.inv_std.cpu()
            vae.scale = [vae.mean, vae.inv_std]

from contextlib import suppress

def _clear_vae_feature_cache(vae_obj):
    """
    Drop any feature caches kept by the VAE or its inner model/decoder.
    Avoid cross-device tensors when we switch cuda:0 -> cuda:1.
    """
    if vae_obj is None:
        return

    # diffusers AutoencoderKLWan sometimes has these
    for attr in ("_feat_map", "_conv_idx"):
        if hasattr(vae_obj, attr):
            with suppress(Exception):
                setattr(vae_obj, attr, None if attr == "_feat_map" else 0)

    # custom WanVAE wrapper -> inner model exposes clear_cache()
    inner = getattr(vae_obj, "model", None)
    if inner is not None and hasattr(inner, "clear_cache"):
        with suppress(Exception):
            inner.clear_cache()

    # also try decoder-level caches (custom implementations)
    dec = getattr(vae_obj, "decoder", None)
    for obj in (vae_obj, inner, dec):
        if obj is None:
            continue
        for attr in ("feat_cache", "_feat_cache", "_feat_map", "_enc_feat_map"):
            if hasattr(obj, attr):
                with suppress(Exception):
                    setattr(obj, attr, None)
        for attr in ("_conv_idx", "_enc_conv_idx"):
            if hasattr(obj, attr):
                with suppress(Exception):
                    setattr(obj, attr, 0)

def _align_vae_constants_device(vae_obj, device: str):
    """
    Some VAE impls keep scale/mean/std as plain attributes (lists/tuples of tensors),
    which .to(device) won't move. Force-move them to the target device.
    """
    if vae_obj is None:
        return

    def _move_attr(obj, name):
        val = getattr(obj, name, None)
        if torch.is_tensor(val):
            setattr(obj, name, val.to(device, non_blocking=True))
        elif isinstance(val, (list, tuple)):
            moved = []
            changed = False
            for x in val:
                if torch.is_tensor(x):
                    moved.append(x.to(device, non_blocking=True)); changed = True
                else:
                    moved.append(x)
            if changed:
                setattr(obj, name, type(val)(moved))

    # common field names across WanVAE / AutoencoderKLWan variants
    for nm in ["scale", "scaler", "scaler_mean", "scaler_std",
               "mean", "std", "mean_tensor", "std_tensor"]:
        _move_attr(vae_obj, nm)

    # inner model/decoder may also host these constants
    for inner_name in ["model", "decoder"]:
        inner = getattr(vae_obj, inner_name, None)
        if inner is not None:
            for nm in ["scale", "scaler", "scaler_mean", "scaler_std",
                       "mean", "std", "mean_tensor", "std_tensor"]:
                _move_attr(inner, nm)

# ===================================================================


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

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1
def dup_first_frame_1cthw_simple(video_tensor):
    return torch.cat([video_tensor[:, :, :1], video_tensor], dim=2)

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

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale(tensor, tW,tH):

    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, tW, tH, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples

def upscale_then_center_crop(img: Image.Image, scale: int, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW, sH = w0 * scale, h0 * scale
    # 先放大
    up = img.resize((sW, sH), Image.BICUBIC)
    # 中心裁剪
    l = max(0, (sW - tW) // 2); t = max(0, (sH - tH) // 2)
    return up.crop((l, t, l + tW, t + tH))

def split_into_segments(total_frames, segment_length=81,):
    """
    将总帧数分割为指定长度的段
    
    Args:
        total_frames: 总帧数 (x)
        segment_length: 每段长度 (默认81)
    
    Returns:
        segments: 包含每段起始和结束索引的列表
    """
    segments = []
    start = 0
    while start < total_frames:
        end = min(start + segment_length, total_frames)
        segments.append((start, end))
        start = end  # 无重叠，直接跳到下一组
    
    return segments

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

def prepare_input_tensor(path, scale: int = 4,fps=30,dtype=torch.bfloat16, device='cuda'):
    if isinstance(path,torch.Tensor):
        total,h0,w0,_ = path.shape
        if total == 1:
            print("got image,repeating to 25 frames")
            path = path.repeat(25, 1, 1, 1) 
            total=25
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        pil_list = tensor2pillist(path)
        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        idx = idx[:F]
        frames = []
        pil_list = [pil_list[i] for i in idx]
        for img in pil_list:                       # ← iterate sublist directly
            img = img.convert('RGB')
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
        print(f"[{os.path.basename(path)}] Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")
        
        frames = []
        for p in paths:
            with Image.open(p) as _im:             # ← fix context manager misuse
                img = _im.convert('RGB')
            img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))   
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)             
        fps = 30
        return vid, tH, tW, F, fps

    elif is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n); n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total - 1] * 4
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
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)   # 1 C F H W
        return vid, tH, tW, F, fps
    else:
        raise ValueError(f"Unsupported input: {path}")

def init_pipeline(prompt_path,LQ_proj_in_path="./FlashVSR/LQ_proj_in.ckpt",ckpt_path: str = "./FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors", vae_path: str = "./FlashVSR/Wan2.1_VAE.pth",decode_vae="none",cur_dir="",device="cuda"):
    #print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    new_decoder=True if decode_vae!="none"  else False
    mm.load_models([ckpt_path,vae_path,])
    pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    
    if new_decoder:
        pipe.new_decoder = True
        if "light" in decode_vae.lower() or "tae" in decode_vae.lower():
            if os.path.basename(decode_vae).split(".")[0]=="lightvaew2_1":
                from ...vae import WanVAE
                print("use lightvae decoder")
                VAE = WanVAE(vae_path=decode_vae,dtype=torch.bfloat16,device=device,use_lightvae=True)
            elif os.path.basename(decode_vae).split(".")[0]=="taew2_1":
                from ...vae_tiny import WanVAE_tiny
                print("use vae_tiny decoder")
                VAE = WanVAE_tiny(vae_path=vae_path,dtype=torch.bfloat16,device=device,need_scaled=False)
            elif os.path.basename(decode_vae).split(".")[0]=="lighttaew2_1":
                from ...vae_tiny import WanVAE_tiny
                print("use vae_tiny light decoder")
                VAE = WanVAE_tiny(vae_path=decode_vae,dtype=torch.bfloat16,device=device,need_scaled=True)
            else:
                raise ValueError(f"Unknown vae_name: {decode_vae},only support lightvae,tae,tae_tiny,lighttae_tiny")
            pipe.VAE=VAE
        else:    
            print("use upscale2x decoder")
            from diffusers import AutoencoderKLWan
            config=AutoencoderKLWan.load_config(os.path.join(cur_dir,"FlashVSR/examples/config.json"))
            VAE=AutoencoderKLWan.from_config(config).to(device,dtype=torch.bfloat16)
            vae_dict=load_file(decode_vae,device="cpu")
            VAE.load_state_dict(vae_dict,strict=False)
            pipe.VAE=VAE
            del vae_dict
    
        
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu",weights_only=False), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path); pipe.load_models_to_device(["dit","vae"])
    return pipe



def run_inference(pipe,input,seed,scale,kv_ratio=3.0,local_range=9,step=1,cfg_scale=1.0,sparse_ratio=2.0,tiled=True,color_fix=True,fix_method="wavelet",split_num=161,dtype=torch.bfloat16,device="cuda",save_vodeo_=False,):
    pipe.to('cuda')  #pipe.enable_vram_management(num_persistent_param_in_dit=None)
    #pipe.init_cross_kv(prompt_path); pipe.load_models_to_device(["dit","vae"])
    _vram = VRAMMeter(devices=("cuda:0","cuda:1"))
    _vram.snap("start")

    pad_first_frame = True  if "wavelet"== fix_method and color_fix else False

    #total,h0,w0,_ = input.shape
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    LQ, th, tw, F, fps = prepare_input_tensor(input, scale=scale, dtype=dtype, device=device)

    frames = pipe(
        prompt="", negative_prompt="", cfg_scale=cfg_scale, num_inference_steps=step, seed=seed, tiled=tiled,
        LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio*768*1280/(th*tw), 
        kv_ratio=kv_ratio,
        local_range=local_range, # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
        color_fix = color_fix,
    )
    pipe.dit.to('cpu')
    torch.cuda.empty_cache()

    tiler_kwargs = {"tiled": tiled, "tile_size": (60, 104), "tile_stride": (30, 52)}
    with torch.no_grad():
        def _decode_on(device_for_decode: str, use_segment_fallback: bool = True):
            # Move only VAE to target device; keep others offloaded
            _move_vae(pipe, device_for_decode)

            # 调用 _move_vae(pipe, device_for_decode) 之后临时加：
            try:
                p = next(pipe.VAE.model.parameters()).device
                print(f"[debug] WanVAE weights device -> {p}, target={device_for_decode}")
            except Exception as _e:
                pass

            # ---- NEW: move latents to target device & clear VAE feature cache ----
            frames_local = frames.to(device_for_decode, non_blocking=True)

            # NEW: drop stale feature caches from previous device
            _clear_vae_feature_cache(getattr(pipe, "VAE", None))
            
            # ★ 新增：把 VAE 的标定常量也同步到目标卡
            _align_vae_constants_device(getattr(pipe, "VAE", None), device_for_decode)

            if hasattr(pipe, "VAE"):
                if hasattr(pipe.VAE, "_feat_map"):
                    pipe.VAE._feat_map = None
                if hasattr(pipe.VAE, "_conv_idx"):
                    with suppress(Exception):
                        pipe.VAE._conv_idx = 0
            # ---------------------------------------------------------------------

            _vram.reset_peak(device_for_decode)
            _vram.snap(f"decode_enter:{device_for_decode}")

            amp_dtype = _amp_dtype()
            try:
                with _amp_cast(amp_dtype):
                    out_full = pipe.decode_video(frames_local, **tiler_kwargs)
                _vram.snap(f"decode_done_once:{device_for_decode}")
                return out_full
            except RuntimeError as e:
                if not (_is_oom_error(e) and use_segment_fallback):
                    raise
                print(f"VAE decode OOM on {device_for_decode}. Falling back to segmented decoding...")

                total_frames = frames_local.shape[2]
                # keep your original heuristic, but CPU-dump each chunk to reduce VRAM peak
                segment_size = max(1, (split_num - 1) * 2 // 4)
                decoded_cpu = []
                for start_idx in range(0, total_frames, segment_size):
                    end_idx = min(start_idx + segment_size, total_frames)
                    seg = frames_local[:, :, start_idx:end_idx, :, :].to(device_for_decode, non_blocking=True)
                    
                    _clear_vae_feature_cache(getattr(pipe, "VAE", None))  # NEW
                    _align_vae_constants_device(getattr(pipe, "VAE", None), device_for_decode)  # ★ 新增


                    with _amp_cast(amp_dtype):
                        dec_seg = pipe.decode_video(seg, **tiler_kwargs)
                    dec_seg = dec_seg.to("cpu", non_blocking=True)  # ← dump to CPU immediately
                    decoded_cpu.append(dec_seg)
                    del seg, dec_seg
                    with suppress(Exception):
                        torch.cuda.empty_cache()
                    _vram.snap(f"seg_post_cpu_dump:{device_for_decode}:{start_idx}-{end_idx}")
                frames_cat = torch.cat(decoded_cpu, dim=2)     # cat on CPU

                _vram.snap("cpu_cat_done")
                return frames_cat

        try:
            # 1) Offload non-decode modules then try on primary device first
            _offload_non_decode_modules(pipe, keep=("VAE","ColorCorrector"))
            frames = _decode_on(device_for_decode=device, use_segment_fallback=False)
        except RuntimeError as e0:
            if not _is_oom_error(e0):
                raise
            # 2) Try secondary GPU (e.g., cuda:1); else segment on primary
            dev1 = _pick_secondary_decode_device(primary_device=device)
            if dev1 is None:
                print("Decode OOM and no secondary GPU. Trying segmented decode on primary...")
                _offload_non_decode_modules(pipe, keep=("VAE","ColorCorrector"))
                frames = _decode_on(device_for_decode=device, use_segment_fallback=True)
            else:
                print(f"Decode OOM on {device}. Retrying on {dev1}...")
                _offload_non_decode_modules(pipe, keep=("VAE","ColorCorrector"))
                frames = _decode_on(device_for_decode=dev1, use_segment_fallback=True)
                # bring back to primary for post-proc
                frames = frames.to(device, non_blocking=True)
        try:
            if color_fix:
                if pad_first_frame:
                    frames = dup_first_frame_1cthw_simple(frames)
                    LQ=dup_first_frame_1cthw_simple(LQ)
                if pipe.new_decoder and LQ.shape[-1]!=frames.shape[-1]:
                    scale_=int(frames.shape[-1]/LQ.shape[-1])
                    LQ=upscale_lq_video_bilinear(LQ,scale_)
                frames = pipe.ColorCorrector(
                    frames.to(device=device),
                    LQ[:, :, :frames.shape[2], :, :],
                    clip_range=(-1, 1),
                    chunk_size=16,
                    method=fix_method
                )
                if pad_first_frame:
                    frames = frames[:, :, 1:, :, :] # remove first frame
        except:
            pass
        print("Done.")
        with suppress(Exception):
            pipe.vae.to('cpu')
    del LQ
    torch.cuda.empty_cache()    
    frames = tensor2video(frames[0])   
    
    if save_vodeo_:
        save_video(frames, os.path.join(folder_paths.get_output_directory(),f"FlashVSR_Full_seed{seed}.mp4"), fps=fps, quality=6)
    _vram.snap("end")
    _vram.to_csv()
    return frames

def upscale_lq_video_bilinear(LQ_video,scale_):
    B, C, T, H, W = LQ_video.shape
    LQ_reshaped = LQ_video.view(B*T, C, H, W) 
    HQ_reshaped = F.interpolate(
        LQ_reshaped, 
        size=(H*scale_, W*scale_),
        mode='bilinear', 
        align_corners=False
    )
    
    HQ_video = HQ_reshaped.view(B, C, T, H*scale_, W*scale_)
    
    return HQ_video
