from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch, io
from PIL import Image

app = FastAPI(title="Qwen-Image-Lightning Background Removal API")

print("Loading model... (Qwen + Lightning)")
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

@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    white_bg: bool = Form(default=False),
    brightness: float = Form(default=1.0)
):
    # 1️⃣ 读取输入图片
    input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # 2️⃣ Prompt：控制模型输出干净背景
    prompt = "product photo with clean white background, professional lighting, isolated subject, no background"
    negative_prompt = "clutter, shadow, watermark, reflection, multiple objects"

    # 3️⃣ 模型推理
    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            num_inference_steps=8,
            width=1024,
            height=1024,
            true_cfg_scale=1.0,
        ).images[0]

    # 4️⃣ 后处理：白底或透明输出
    if white_bg:
        bg = Image.new("RGB", result.size, (255, 255, 255))
        bg.paste(result, mask=result.split()[3] if result.mode == "RGBA" else None)
        output = bg
    else:
        output = result.convert("RGBA")

    # 5️⃣ 亮度增强（简单线性）
    if brightness != 1.0:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(output)
        output = enhancer.enhance(brightness)

    # 6️⃣ 输出图片流
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

@app.get("/")
def root():
    return {"message": "Qwen-Image-Lightning Background Removal API Ready"}
