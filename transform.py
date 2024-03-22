import torch
import boto3
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

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
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name
object_name = 'input_image.jpg'  # Replace with your input image name
response = s3.get_object(Bucket=bucket_name, Key=object_name)
image_bytes = response['Body'].read()

# Open the initial image
init_image = Image.open(BytesIO(image_bytes)).convert("RGB")

# Define the prompt
prompt = "add a realistic  rainbow "

# Generate the image
image = pipe(prompt, image=init_image).images[0]

# Convert the PIL image to bytes
image_byte_array = BytesIO()
image.save(image_byte_array, format='JPEG')
image_bytes = image_byte_array.getvalue()

# Upload the image to S3
object_name = 'generated_image.jpg'  # Replace with your desired object name
response = s3.put_object(Bucket=bucket_name, Key=object_name, Body=image_bytes)

print(f"Image uploaded to S3 bucket {bucket_name} with key {object_name}")