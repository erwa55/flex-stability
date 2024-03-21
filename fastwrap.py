from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline
import boto3
from io import BytesIO

# Define a request model
class ImagePrompt(BaseModel):
    prompt: str

# Initialize FastAPI app
app = FastAPI()

# Initialize the pipeline (do this outside of your endpoint to avoid re-initialization)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Initialize S3 client (do this outside of your endpoint to avoid re-initialization)
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name

# Custom exception handler as middleware
@app.middleware("http")
async def exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # Log the exception
        logger.exception(f"Unhandled error: {e}")
        # Return a generic response to the client
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."}
        )
@app.post("/generate-image/")
async def generate_image(prompt: ImagePrompt):
    try:
        # Generate the image with the received prompt
        images = pipe(prompt=prompt.prompt).images[0]

        # Convert the PIL image to bytes
        image_byte_array = BytesIO()
        images.save(image_byte_array, format='JPEG')  # You can change 'JPEG' to 'PNG' if you prefer
        image_bytes = image_byte_array.getvalue()

        # Construct a unique object name, for example using a timestamp or a UUID
        from datetime import datetime
        object_name = f"image-{datetime.utcnow().isoformat()}.jpg"

        # Upload to S3
        response = s3.put_object(Bucket=bucket_name, Key=object_name, Body=image_bytes)

        return {"message": f"Image successfully uploaded to S3 bucket {bucket_name} with key {object_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

