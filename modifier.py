import PIL
import requests
import torch
import boto3
from io import BytesIO
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Initialize the Stable Diffusion pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Function to download the image
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Download the example image
url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
image = download_image(url)

# Generate the image with the specified prompt
prompt = "turn him into cyborg"
generated_images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

# Convert the PIL image to bytes for upload
img_byte_arr = BytesIO()
generated_images[0].save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

# Setup boto3 client
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name
object_name = 'cyborg_transformation.jpg'  # Desired object name in S3

# Upload the image to S3
s3.put_object(Bucket=bucket_name, Key=object_name, Body=img_byte_arr)

print(f"Image successfully uploaded to s3://{bucket_name}/{object_name}")
