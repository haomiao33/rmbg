

import io
import torch
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import runpod  # 来自 RunPod SDK
import requests

# 初始化模型（只做一次）
scheduler = FlowMatchEulerDiscreteScheduler.from_config("Qwen/Qwen-Image")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

def handler(event):
    input_data = event.get("input", {})
    image_url = input_data.get("image_url")
    if not image_url:
        return {"error": "Missing image_url in input"}

    resp = requests.get(image_url)
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    with torch.autocast("cuda"):
        result = pipe(
            prompt="product photo with clean isolated subject, no background",
            negative_prompt="background, clutter, shadows, multiple objects",
            image=image,
            num_inference_steps=8,
            width=1024,
            height=1024,
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    import base64
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    return {"image_base64": encoded, "status": "success"}

# 启动服务（RunPod 要求）
runpod.serverless.start({"handler": handler})
