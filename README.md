# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI

# Upadte
* 增加了对双GPU的支持,当cuda0在decode阶段OOM时触发由cuda1来decode,可以适当的增加上限.
# Tips
*  仅修改了full模式和tiny模式,没有对tiny-long进行适配.
*  根据多GPU的逻辑优化了OOM的流程,提升了上限.
*  Block-Sparse-Attention 编译,修改了setup.py让编译更容易.  
  
1.Installation  
-----
  A) Intel XPU 版本（XPU_v1.0）   
  在ComfyUI\custom_nodes\目录下运行:
```
git clone --branch XPU_v1.0 --single-branch https://github.com/Goldlionren/ComfyUI_FlashVSR.git

```
  B) CUDA 双卡版本（cuda_dual_gpu_v1）   
  在ComfyUI\custom_nodes\目录下运行:
```
git clone --branch cuda_dual_gpu_v1 --single-branch https://github.com/Goldlionren/ComfyUI_FlashVSR.git

```

2.requirements  
----

```
pip install -r requirements.txt
```
要复现官方效果，必须安装Block-Sparse-Attention Pytorch=2.7和2.8的朋友们可以使用大神们的:[torch2.8 cu2.8 py311 wheel ](https://pan.quark.cn/s/c9ba067c89bc) or [CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper),如果是其他版本的,请根据以下流程进行:
```
git clone https://github.com/mit-han-lab/Block-Sparse-Attention 
# git clone https://github.com/smthemex/Block-Sparse-Attention # 无须梯子强制编译
# git clone https://github.com/lihaoyun6/Block-Sparse-Attention # 须梯子
cd Block-Sparse-Attention
pip install packaging
pip install ninja
```
然后查看并修改setup.py,把里面的显卡代号修改成自己需要的,类似如下:
```
 cc_flag.append("-gencode")
 cc_flag.append("arch=compute_80,code=sm_80")
 if CUDA_HOME is not None:
     if bare_metal_version >= Version("11.8"):
         cc_flag.append("-gencode")
         cc_flag.append("arch=compute_90,code=sm_90")
+        # Ada / Lovelace (SM 8.9)
+        cc_flag.append("-gencode")
+        cc_flag.append("arch=compute_89,code=sm_89")
+        # Ampere consumer (SM 8.6)，如 3080/3090
+        cc_flag.append("-gencode")
+        cc_flag.append("arch=compute_86,code=sm_86")
+        # 可选：嵌入 PTX，提升前向兼容
+        cc_flag.append("-gencode")
+        cc_flag.append("arch=compute_90,code=compute_90")

```
修改后就可以编译了:
```
python.exe setup.py build_ext -v install
```


3.checkpoints 
----

* 3.1 [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1  
* 3.2 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
* 3.3 [lightvaew2_1.pth](https://huggingface.co/lightx2v/Autoencoders/tree/main) and [diffusion_pytorch_model.safetensors](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/tree/main/diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1)
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── posi_prompt.pth
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth
|        ├──lightvaew2_1.pth  #32.2M  or taew2_1.pth,lighttaew2_1.pth
|        ├──Wan2.1_VAE_upscale2x_imageonly_real_v1_diff.safetensors  # rename from diffusion_pytorch_model.safetensors
```
  
