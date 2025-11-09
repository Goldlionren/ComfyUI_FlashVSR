# 🚀 ComfyUI FlashVSR (Goldlionren Edition)

> Enhanced version of [smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR)  
> by [Goldlionren](https://github.com/Goldlionren), featuring **dual-GPU CUDA support**, **multi-device safe memory logic**, and **v11 compatibility**.

---

## ✨ What's New (2025-11)

✅ **Dual-GPU CUDA decoding**  
- Added logic for automatic fallback between `cuda:0` and `cuda:1`  
- Prevented cross-device tensor mismatches  
- Ensures stable decoding even when out of GPU memory  

✅ **Improved TCDecoder memory logic**  
- Safe cross-device memory reuse  
- Adds CPU fallback for out-of-memory stacking  
- Eliminates device mismatch crashes  

✅ **v11-compatible infer scripts**  
- Updated:  
  - `infer_flashvsr_v11_full.py`  
  - `infer_flashvsr_v11_tiny.py`  
- Based on upstream `v11` structure but retains Goldlionren’s multi-GPU logic  

✅ **Clean upstream sync (Nov 2025)**  
- Integrated the latest changes from `smthemex/main`  
- Clean rebase and code merge with conflict resolution  

---

## 🔧 Installation

Clone the latest **dual-GPU** branch:

```bash
git clone -b cuda_dual_gpu_v1_1 --single-branch https://github.com/Goldlionren/ComfyUI_FlashVSR.git
```

For Intel XPU users (Arc / iGPU), use the `XPU_v1.0` branch:

```bash
git clone -b XPU_v1.0 --single-branch https://github.com/Goldlionren/ComfyUI_FlashVSR.git
```

Then place the folder under your ComfyUI `custom_nodes` directory:
```
ComfyUI/custom_nodes/ComfyUI_FlashVSR
```

---

## ⚙️ Requirements

- Python ≥ 3.10  
- PyTorch ≥ 2.7.0 + CUDA 12.4  
- Compatible with:
  - NVIDIA RTX 40xx Series  
  - Dual-GPU systems (tested on RTX 4080 SUPER + 4060 Ti)  
  - Intel XPU (Arc A770 / A770M) via `XPU_v1.0` branch  

---

## 📂 Branch Summary

| Branch | Description |
|--------|--------------|
| `main` | Upstream base (synced with smthemex) |
| `XPU_v1.0` | Intel XPU build (Arc A770 / A770M) |
| `cuda_dual_gpu_v1_1` | NVIDIA dual-GPU build (v11 integration, stable) |

---

## 🧠 Technical Notes

- `TCDecoder.py` now includes robust device safety:
  - Automatically reinitializes tensors when device mismatch detected  
  - Supports CPU fallback when GPU memory exhausted  
  - Ensures consistent decode state across re-runs  

- `infer_flashvsr_full.py` / `infer_flashvsr_tiny.py` now:
  - Detect GPU availability dynamically  
  - Retry decode on secondary device  
  - Integrate ComfyUI prompt parameters for flexible inference  

---

## 🇨🇳 中文说明

本分支为在 ComfyUI 上运行 FlashVSR 的 **增强版**，主要改进包括：

- **双显卡自动分担解码任务**（CUDA 版本）  
- **显存不足自动回退机制**  
- **跨设备安全内存管理（TCDecoder 修复）**  
- **兼容 FlashVSR v11 的新版推理脚本**  
- 已同步原作者仓库 2025 年 11 月更新内容  

---

## 🧩 Maintainer

**Goldlionren**  
📦 Repo: [https://github.com/Goldlionren/ComfyUI_FlashVSR](https://github.com/Goldlionren/ComfyUI_FlashVSR)
