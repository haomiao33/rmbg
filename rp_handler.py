import io
import os
import base64
import torch
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import runpod
import requests

# 读取模型路径（RunPod 缓存）
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen-Image")
LORA_PATH = os.getenv("LORA_PATH", "lightx2v/Qwen-Image-Lightning")
LORA_WEIGHT = os.getenv("LORA_WEIGHT", "Qwen-Image-Lightning-8steps-V1.0.safetensors")

# 初始化模型，只加载一次（在容器启动时）
print(f"🔹 Loading model from: {MODEL_PATH}")
scheduler = FlowMatchEulerDiscreteScheduler.from_config(MODEL_PATH)
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)
print(f"🔹 Loading LoRA: {LORA_PATH}")
pipe.load_lora_weights(LORA_PATH, weight_name=LORA_WEIGHT)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

# 主处理函数
def handler(event):
    input_data = event.get("input", {})
    image_url = input_data.get("image_url")

    if not image_url:
        return {"error": "Missing image_url in input"}

    # 下载输入图像
    resp = requests.get(image_url)
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 生成清理后的图像
    with torch.autocast("cuda"):
        result = pipe(
            prompt="product photo with clean isolated subject, no background",
            negative_prompt="background, clutter, shadows, multiple objects",
            image=image,
            num_inference_steps=8,
            width=1024,
            height=1024,
        ).images[0]

    # 转 base64
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"status": "success", "image_base64": encoded}

# RunPod Serverless 启动
runpod.serverless.start({"handler": handler})
