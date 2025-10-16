import io
import torch
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import runpod  # ✅ Required

# -------------------------
# 1️⃣ 模型初始化（只加载一次）
# -------------------------
print("Loading Qwen-Image-Lightning model...")

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

# -------------------------
# 2️⃣ 事件处理函数
# -------------------------
def handler(event):
    """
    event: dict with {"input": {"image_url": "https://...", ...}}
    """
    input_data = event.get("input", {})
    image_url = input_data.get("image_url")

    if not image_url:
        return {"error": "Missing 'image_url' in input"}

    # 下载输入图片
    import requests
    resp = requests.get(image_url)
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 模型推理
    with torch.autocast("cuda"):
        result = pipe(
            prompt="product photo with clean white background, isolated subject",
            negative_prompt="clutter, shadow, background, watermark",
            image=image,
            num_inference_steps=8,
            width=1024,
            height=1024,
        ).images[0]

    # 保存到 buffer
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    # RunPod serverless 返回需要 base64
    import base64
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    return {"image_base64": encoded, "status": "success"}

# -------------------------
# 3️⃣ 启动 RunPod Serverless
# -------------------------
runpod.serverless.start({"handler": handler})
