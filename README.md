# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI

# Upadte
*  满足部分网友需要超分单张图片的奇怪要求,默认输出25帧1秒的视频，详见示例，Block-Sparse-Attention 目前不支持5090的sm120架构，需要改一下Block-Sparse-Attention的源码来支持； 
*  同步tiny的专属long模式  
*  新增切片视频路径加载节点，输入保存切片视频的路径，开启自动推理，即可推理完路径所有视频； 
*  修复输入图像归一化处理错误导致无法复现官方的问题，分离decoder，新增关键点模型卸载和OOM处理，包括处理超长视频向量的OOM，同步官方local range的修改，新增小波模式下的加减帧处理（项目一作大佬提的）；
*  local_range=7这个是会最清晰，local_range=11会比较稳定，color fix 推荐用小波（没重影）； 
*  编译Block-Sparse-Attention  window的轮子 可以使用 [ smthemex 强制编译版](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6 要联网](https://github.com/lihaoyun6/Block-Sparse-Attention) 两个fork来，不推荐用官方的  
*  Block-Sparse-Attention 正确安装且能调用才是方法的完全体，当前的函数实现会更容易OOM,但是Block-Sparse-Attention轮子实在不好找，目前只有[CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper)的，我提供的（[cu128，torch2.8，py311单体](https://pan.quark.cn/s/c9ba067c89bc)）或者自己编译  
*  方法是基于现有prompt.pt训练的，所以外置cond没有必要已经去掉，新增tile 和 color fix 选项，tile关闭质量更高，需要VRam更高，corlor fix对于非模糊图片可以试试。修复图片索引数不足的错误。  
*  Choice vae infer full mode ，encoder infer tiny mode 选择vae跑full模式 效果最好，tiny则是速度，数据集基于4倍训练，所以1 scale是不推荐的；  
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 打星； if you Like it ， star the official project [link](https://github.com/OpenImagingLab/FlashVSR)

  
1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FlashVSR

```

2.requirements  
----

```
pip install -r requirements.txt
```
要复现官方效果，必须安装Block-Sparse-Attention [torch2.8 cu2.8 py311 wheel ](https://pan.quark.cn/s/c9ba067c89bc) or [CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper)
```
git clone https://github.com/mit-han-lab/Block-Sparse-Attention 
# git clone https://github.com/smthemex/Block-Sparse-Attention # 无须梯子强制编译
# git clone https://github.com/lihaoyun6/Block-Sparse-Attention # 须梯子
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

3.checkpoints 
----

* 3.1 [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1  
* 3.2 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── posi_prompt.pth
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth
```
  

# Example
* single image VSR 
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example_s.png)
* full old node 
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example18.png)
* tiny new
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example1022.png)
* video files loop
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example.png)

# Acknowledgements
[DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)  
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)  
[taehv](https://github.com/madebyollin/taehv)  

# Citation
```
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}

``

``
