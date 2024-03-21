import torch
import boto3
from io import BytesIO
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# Initialize the pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Load the initial image from the URL
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
init_image = load_image(url).convert("RGB")

# Define the prompt
prompt = "a photo of an astronaut riding a horse on mars"

# Generate the image
image = pipe(prompt, image=init_image).images[0]

# Convert the PIL image to bytes
image_byte_array = BytesIO()
image.save(image_byte_array, format='JPEG')
image_bytes = image_byte_array.getvalue()

# Upload the image to S3
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name
object_name = 'generated_image.jpg'  # Replace with your desired object name
response = s3.put_object(Bucket=bucket_name, Key=object_name, Body=image_bytes)

print(f"Image uploaded to S3 bucket {bucket_name} with key {object_name}")