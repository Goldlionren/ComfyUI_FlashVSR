 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  tensor_upscale,load_images_list,get_video_files
from .FlashVSR.examples.WanVSR.infer_flashvsr_full import init_pipeline,run_inference
from .FlashVSR.examples.WanVSR.infer_flashvsr_tiny import   init_pipeline_tiny,run_inference_tiny
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from pathlib import PureWindowsPath
from comfy_api.input_impl import VideoFromFile

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_FlashVSR_current_path = os.path.join(folder_paths.models_dir, "FlashVSR")
if not os.path.exists(weigths_FlashVSR_current_path):
    os.makedirs(weigths_FlashVSR_current_path)

folder_paths.add_model_folder_path("FlashVSR", weigths_FlashVSR_current_path) #  FlashVSR dir


class FlashVSR_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="FlashVSR_SM_Model",
            display_name="FlashVSR_SM_Model",
            category="FlashVSR",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "dmd" in i.lower()]),
                io.Combo.Input("proj_pt",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "proj" in i.lower()]),
                io.Combo.Input("emb_pt",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "prompt" in i.lower()]),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("tcd_encoder",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "tcd" in i.lower()] ),
            ],
            outputs=[
                io.Custom("FlashVSR_SM_Model").Output(),
                ],
            )
    @classmethod
    def execute(cls, dit,proj_pt,emb_pt,vae,tcd_encoder) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("FlashVSR", dit) if dit != "none" else None
        proj_pt_path=folder_paths.get_full_path("FlashVSR", proj_pt) if proj_pt != "none" else None
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        tcd_encoder_path=folder_paths.get_full_path("FlashVSR", tcd_encoder) if tcd_encoder != "none" else None
        prompt_path=folder_paths.get_full_path("FlashVSR", emb_pt) if emb_pt != "none" else None
        assert prompt_path is not None  , "Please select the emb"
        assert dit_path is not None and proj_pt is not None , "Please select the Sdit,proj_pt,checkpoint file"
        assert vae_path is not None or tcd_encoder_path is not None , "Please select the Sdit,proj_pt,checkpoint file"
        if vae_path is None and tcd_encoder_path is not None:
            model=init_pipeline_tiny(prompt_path,proj_pt_path,dit_path, tcd_encoder_path, device="cuda")
        else:
            model=init_pipeline(prompt_path,proj_pt_path,dit_path, vae_path, device="cuda")
        return io.NodeOutput(model)
    

class FlashVSR_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_KSampler",
            display_name="FlashVSR_SM_KSampler",
            category="FlashVSR",
            inputs=[
                io.Custom("FlashVSR_SM_Model").Input("model"),
                io.Image.Input("image"),
                io.Int.Input("width", default=1280, min=128, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=768, min=128, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("scale", default=4, min=1, max=4),
                io.Float.Input("kv_ratio", default=3.5, min=0.0, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("local_range", default=11, min=1,step=1, max=50),
                io.Int.Input("steps", default=1, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Float.Input("sparse_ratio", default=2.0, min=0.0, max=10.0, step=0.1,), 
                io.Boolean.Input("full_tiled", default=True),
                io.Boolean.Input("color_fix", default=True),
                io.Combo.Input("fix_method",options= ["wavelet","adain"]),
                io.Int.Input("split_num", default=81, min=41, max=MAX_SEED,step=40,),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )
    @classmethod
    def execute(cls, model,image,width,height,seed,scale,kv_ratio,local_range, steps, cfg,sparse_ratio,full_tiled,color_fix,fix_method,split_num) -> io.NodeOutput:
        image=tensor_upscale(image,width, height)
       
        if hasattr(model,"TCDecoder") :
            print("infer tiny mode")
            images=run_inference_tiny(model,image,seed,scale,kv_ratio,local_range,steps,cfg,sparse_ratio,color_fix,fix_method,split_num )
        else:
            print("infer full mode")
            images=run_inference(model,image,seed,scale,kv_ratio,local_range,steps,cfg,sparse_ratio,full_tiled,color_fix,fix_method,split_num )
        images=load_images_list(images)
        return io.NodeOutput(images)


class FlashVSR_SM_VideoPathLoop(io.ComfyNode):
    @classmethod
    def __init__(cls):
        cls.counters = {}
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_VideoPathLoop",
            display_name="FlashVSR_SM_VideoPathLoop",
            category="FlashVSR",
            inputs=[
                io.String.Input("video_dir", multiline=False, default="/video"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("start", default=0.0, min=-18446744073709551615, max=18446744073709551615, step=0.01,),
                io.Float.Input("stop", default=0.0, min=-18446744073709551615, max=18446744073709551615, step=0.01,),
                io.Float.Input("step", default=1, min=0,max=99999,step=0.01, ),
                io.Combo.Input("mode",options= ["increment", "decrement", "increment_to_stop", "decrement_to_stop"],),
                io.Combo.Input("video_file", options=['none', 'webm', 'mp4', 'mkv', 'gif', 'mov']),
                io.Custom("NUMBER").Input("reset_bool",optional=True),
            ],
            outputs=[
                io.Video.Output(),
                io.Custom("NUMBER").Output(display_name="number"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="filename"),
            ],
        )
    
    @classmethod
    def execute(cls, video_dir,seed, mode, start, stop, step,video_file,reset_bool=0,**kwargs) -> io.NodeOutput:
        video_path = PureWindowsPath(video_dir).as_posix() if video_dir else None
        video_file = None if video_file == 'none' else video_file
        assert video_path is not None, "video_dir is not set"
        UNIQUE_ID = os.path.normpath(video_path) 
        counter =start
        if cls.counters.__contains__(UNIQUE_ID):
            counter = cls.counters[UNIQUE_ID]
        if round(reset_bool) >= 1:
            counter = start

        if mode == 'increment':
            counter += step
        elif mode == 'decrement':
            counter -= step
        elif mode == 'increment_to_stop':
            counter = counter + step if counter < stop else counter
        elif mode == 'decrement_to_stop':
            counter = counter - step if counter > stop else counter

        cls.counters[UNIQUE_ID] = counter
        result = int(counter)

        
        video_list = get_video_files(video_path, video_file)
        rows = len(video_list) if video_list else 0
        if rows == 0:
            assert False, "no video found"

        if result == 0:
            selected_path = video_list[0]
        else:
            adjusted_index = (result - 1) % rows
            selected_path = video_list[adjusted_index]
            
        print(f"Selected video path: {selected_path}")
        filename=os.path.basename(selected_path)
        return io.NodeOutput(VideoFromFile(selected_path),result, seed,filename)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
       return ""


from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/FlashVSR_SM_Extension")
async def get_hello(request):
    return web.json_response("FlashVSR_SM_Extension")

class FlashVSR_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FlashVSR_SM_Model,
            FlashVSR_SM_KSampler,
            FlashVSR_SM_VideoPathLoop,
        ]
async def comfy_entrypoint() -> FlashVSR_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return FlashVSR_SM_Extension()



