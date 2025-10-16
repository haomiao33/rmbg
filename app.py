from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch, io
from PIL import Image

app = FastAPI(title="Qwen-Image-Lightning Background Removal API")

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

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    with torch.autocast("cuda"):
        result = pipe(
            prompt="a product photo with clean white background, isolated subject, no shadow",
            negative_prompt="blurry, artifact, messy background, low quality",
            image=image,
            num_inference_steps=8,
            width=1024,
            height=1024,
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/")
def root():
    return {"message": "Qwen-Image-Lightning Background Removal API Ready"}
