from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline
import boto3
from io import BytesIO

class ImageRequest(BaseModel):
    prompt: str
    bucket_name: str  # Bucket name where the image will be uploaded
    image_key: str  # The key for the generated image to be saved as in the S3 bucket

# Initialize FastAPI app
app = FastAPI()

# Initialize the pipeline (do this outside of your endpoint to avoid re-initialization)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Initialize S3 client (do this outside of your endpoint to avoid re-initialization)
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

@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    try:
        # Generate the image with the received prompt
        images = pipe(prompt=request.prompt).images[0]

        # Convert the PIL image to bytes
        image_byte_array = BytesIO()
        images.save(image_byte_array, format='JPEG')  # You can change 'JPEG' to 'PNG' if you prefer
        image_bytes = image_byte_array.getvalue()

        # Use the provided bucket name and image key
        response = s3.put_object(Bucket=request.bucket_name, Key=request.image_key, Body=image_bytes)

        return {"message": f"Image successfully uploaded to S3 bucket {request.bucket_name} with key {request.image_key}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
