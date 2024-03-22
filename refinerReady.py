from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import torch
import boto3
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

app = FastAPI()

# Define the request model
class ImageRequest(BaseModel):
    bucket_name: str
    image_key: str
    prompt: str
    generated_image_key: str

# Create the API endpoint
@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    # Initialize the pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe = pipe.to("cuda")

    # Download the initial image from S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=request.bucket_name, Key=request.image_key)
    image_bytes = response['Body'].read()

    # Open the initial image
    init_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Generate the image
    image = pipe(request.prompt, image=init_image).images[0]

    # Convert the PIL image to bytes
    image_byte_array = BytesIO()
    image.save(image_byte_array, format='JPEG')
    image_bytes = image_byte_array.getvalue()

    # Upload the image to S3
    s3.put_object(Bucket=request.bucket_name, Key=request.generated_image_key, Body=image_bytes)

    return {
        "message": "Image uploaded successfully",
        "url": f"s3://{request.bucket_name}/{request.generated_image_key}"
    }
