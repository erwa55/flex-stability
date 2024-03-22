import PIL
import torch
import boto3
from io import BytesIO
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Initialize the Stable Diffusion pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Setup boto3 client
s3 = boto3.client('s3')

# Function to download the image from S3
def download_image_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    image_data = obj['Body'].read()
    image = PIL.Image.open(BytesIO(image_data))
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Specify your S3 bucket and the key of the image you want to download
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your S3 bucket name
image_key = 'input_image.jpg'  # Replace with the key of your image in the S3 bucket

# Download the image from S3
image = download_image_from_s3(bucket_name, image_key)

# Generate the image with the specified prompt
prompt = "add a rainbow in the background"
generated_images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

# Convert the PIL image to bytes for S3 upload
img_byte_arr = BytesIO()
generated_images[0].save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

# Define a new key for the generated image to be saved in S3
generated_image_key = 'generated_image.jpg'

# Upload the generated image to S3
s3.put_object(Bucket=bucket_name, Key=generated_image_key, Body=img_byte_arr)

print(f"Generated image uploaded to s3://{bucket_name}/{generated_image_key}")
