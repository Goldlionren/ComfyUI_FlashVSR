# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI

# Upadte

* 增加了对双GPU的支持,当cuda0在decode阶段OOM时触发由cuda1来decode,可以适当的增加上限.
# Tips
*  仅修改了full模式和tiny模式,没有对tiny-long进行适配.
*  根据多GPU的逻辑优化了OOM的流程,提升了上限.
*  Block-Sparse-Attention 编译,修改了setup.py让编译更容易.  


* update to version v1.1 /更新适配1.1版本的新模型和代码,降低闪烁，提高保真度和稳定性
* add full mode [lightx2v vae encoder](https://huggingface.co/lightx2v/Autoencoders/tree/main) support（only lightvaew2_1.pth,taew2_1.pth,lighttaew2_1.pth） and [Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) support    
* 新增lightx2v 加速vae decoder支持和Wan2.1-VAE-upscale2x 放大decoder支持，只是在full 模式下有效，light的加速模型目前只支持（lightvaew2_1.pth  #32.2M,taew2_1.pth,lighttaew2_1.pth） 三个文件 
# Tips
*  满足部分网友需要超分单张图片的奇怪要求,默认输出25帧1秒的视频，详见示例，Block-Sparse-Attention 目前不支持5090的sm120架构，需要改一下Block-Sparse-Attention的源码来支持； 
*  同步tiny的专属long模式  
*  新增切片视频路径加载节点，输入保存切片视频的路径，开启自动推理，即可推理完路径所有视频； 
*  修复输入图像归一化处理错误导致无法复现官方的问题，分离decoder，新增关键点模型卸载和OOM处理，包括处理超长视频向量的OOM，同步官方local range的修改，新增小波模式下的加减帧处理（项目一作大佬提的）；
*  local_range=7这个是会最清晰，local_range=11会比较稳定，color fix 推荐用小波（没重影）； 
*  编译Block-Sparse-Attention  window的轮子 可以使用 [ smthemex 强制编译版](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6 要联网](https://github.com/lihaoyun6/Block-Sparse-Attention) 两个fork来，不推荐用官方的  
*  Block-Sparse-Attention 正确安装且能调用才是方法的完全体，当前的函数实现会更容易OOM,但是Block-Sparse-Attention轮子实在不好找，目前只有[CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper)的，我提供的（[cu128，torch2.8，py311单体](https://pan.quark.cn/s/c9ba067c89bc)）或者自己编译  
*  方法是基于现有prompt.pt训练的，新增tile 和 color fix 选项，tile关闭质量更高，需要VRam更高，corlor fix对于非模糊图片可以试试。修复图片索引数不足的错误。  
*  Choice vae infer full mode ，encoder infer tiny mode 选择vae跑full模式 效果最好，tiny则是速度，数据集基于4倍训练，所以1 scale是不推荐的；  
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 打星； if you Like it ， star the official project [link](https://github.com/OpenImagingLab/FlashVSR)

  
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

* 3.1.2 [FlashVSRv1.0](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1
* 3.1.2 [FlashVSRv1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/tree/main) all checkpoints 所有模型，vae 用常规的wan2.1
* 3.2 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
* 3.3 [lightvaew2_1.pth](https://huggingface.co/lightx2v/Autoencoders/tree/main) and [diffusion_pytorch_model.safetensors](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/tree/main/diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1)
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt # v1.1 or v1.0
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors #v1.1 or v1.0
|     ├── posi_prompt.pth
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth
|        ├──lightvaew2_1.pth  #32.2M  or taew2_1.pth,lighttaew2_1.pth
|        ├──Wan2.1_VAE_upscale2x_imageonly_real_v1_diff.safetensors  # rename from diffusion_pytorch_model.safetensors
```
  
