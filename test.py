from diffusers import DiffusionPipeline
import torch
import boto3
from io import BytesIO

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Generate the image
prompt = "An astronaut riding a green horse"
images = pipe(prompt=prompt).images[0]

# Convert the PIL image to bytes
image_byte_array = BytesIO()
images.save(image_byte_array, format='JPEG')  # You can change 'JPEG' to 'PNG' if you prefer
image_bytes = image_byte_array.getvalue()

# Upload to S3
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name
object_name = 'your-object-name.jpg'  # Replace with your desired object name

response = s3.put_object(Bucket=bucket_name, Key=object_name, Body=image_bytes)

print(f"Image uploaded to S3 bucket {bucket_name} with key {object_name}")
