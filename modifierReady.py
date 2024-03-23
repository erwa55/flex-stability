from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PIL
import torch
import boto3
from io import BytesIO
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class ImageRequest(BaseModel):
    bucket_name: str
    image_key: str
    prompt: str
    generated_image_key: str

# Initialize FastAPI app
app = FastAPI()

# Initialize the Stable Diffusion pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


# Setup boto3 client globally
s3 = boto3.client('s3')



# Exception handler as middleware
@app.middleware("http")
async def exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # Log the exception here with your preferred logging setup
        print(f"Unhandled error: {e}")  # Example placeholder for logging
        # Return a generic response to the client
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."}
        )

@app.post("/generate-image")
async def generate_image(request: ImageRequest):

    
    # Function to download the image from S3
    async def download_image_from_s3(bucket, key):
        obj = s3.get_object(Bucket=bucket, Key=key)
        image_data = obj['Body'].read()
        image = PIL.Image.open(BytesIO(image_data))
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    # Download the image from S3
    image = await download_image_from_s3(request.bucket_name, request.image_key)

    # Generate the image with the specified prompt
    generated_images = pipe(request.prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

    # Convert the PIL image to bytes for S3 upload
    img_byte_arr = BytesIO()
    generated_images[0].save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Upload the generated image to S3
    s3.put_object(Bucket=request.bucket_name, Key=request.generated_image_key, Body=img_byte_arr)

    return {
        "message": "Generated image uploaded successfully",
        "url": f"s3://{request.bucket_name}/{request.generated_image_key}"
    }
