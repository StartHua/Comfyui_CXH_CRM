import os
import torch
from PIL import Image
import folder_paths
import latent_preview
import numpy as np
import safetensors.torch
import cv2

from PIL import Image, ImageOps, ImageSequence, ImageFile,UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
from omegaconf import OmegaConf

from .lib.ximg import *
from .lib.xmodel import *

import sys
device = "cuda" if torch.cuda.is_available() else "cpu"

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir))

class CRM:
    
    def __init__(self):
        self.pipeline = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "image": ("IMAGE",),
               #    "model": (["CRM", "ImageDream"],),
               "guidance_scale":("FLOAT", {"default": 3, "min": 1, "max": 10, "step": 0.01}),
               "steps":("INT", {"default": 50, "min": 20, "max": 100, "step": 0.01}),
               "seed": ("INT", {"default": 0, "min": 0, "max": 99999999}),       
            }
        }

    CATEGORY = "CXH/3D"
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "load_image"
    
    #  def load_image(self,image,model,guidance_scale,steps,seed):
    def load_image(self,image,guidance_scale,steps,seed):
        
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.step = steps
        
        results = []
        
        #下载模型
        craft_model_id = "Zhengyi/CRM"
        model_checkpoint = download_hg_model(craft_model_id,"CRM")
        ckpt_path= os.path.join(model_checkpoint,"pixel-diffusion.pth") 
        
        from pipelines import TwoStagePipeline
        
        stage1_config = OmegaConf.load(f"{dir}/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
        stage1_sampler_config = stage1_config.sampler
        stage1_model_config = stage1_config.models
        
        stage1_model_config.resume = ckpt_path
        
        stage1_model_config.config = f"{dir}/" + stage1_model_config.config
        
        if self.pipeline == None:
            self.pipeline = TwoStagePipeline(
                                stage1_model_config,
                                stage1_sampler_config,
                                device=device,
                                dtype=torch.float16
                        )
        
 
        self.pipeline.set_seed(self.seed)
        
        rt_dict = self.pipeline(tensor2pil(image), scale=self.guidance_scale, step=self.step)
        mv_imgs = rt_dict["stage1_images"]
        
        mv_imgs[5].save("front.png")
        mv_imgs[3].save("right.png")
        mv_imgs[2].save("back.png")
        mv_imgs[0].save("left.png")

        front = pil2tensor(mv_imgs[5])
        right = pil2tensor(mv_imgs[3])
        back = pil2tensor(mv_imgs[2])
        left = pil2tensor(mv_imgs[0])
            
        results.append(front)
        results.append(right)
        results.append(back)
        results.append(left)
        
        return (torch.cat(results, dim=0),)

