from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionInpaintPipeline
import boto3
from io import BytesIO
from PIL import Image

# ... (the rest of the code remains the same) ...

# Initialize the pipeline (do this outside of your endpoint to avoid re-initialization)
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# ... (the rest of the code remains the same) ...

@app.post("/generate-image/")
async def generate_image(prompt: ImagePrompt = Body(...)):
    try:
        # Fetch the input image from S3
        image_s3_path = prompt.image_s3_path
        bucket, key = image_s3_path.split("/", 1)
        response = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Generate the modified image with the received prompt and image
        images = pipe(prompt=prompt.prompt, image=image).images[0]

        # ... (the rest of the code remains the same) ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))