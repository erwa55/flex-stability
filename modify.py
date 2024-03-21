import DiffusionPipeline
import torch
import boto3
from io import BytesIO
from PIL import Image

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, revision="fp16")
pipe.to("cuda")

# Get the prompt from the user
prompt = input("Add a rainbow ")

# Download the existing image from S3
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name
object_name = 'your-object-name.jpg'  # Replace with your existing image name
response = s3.get_object(Bucket=bucket_name, Key=object_name)
image_bytes = response['Body'].read()

# Open the image
image = Image.open(BytesIO(image_bytes))

# Modify the image based on the prompt
images = pipe(prompt=prompt, image=image, mask_image=None)

# Convert the modified PIL image to bytes
image_byte_array = BytesIO()
images[0].save(image_byte_array, format='JPEG')
modified_image_bytes = image_byte_array.getvalue()

# Upload the modified image to S3
object_name = f"modified_{object_name}"  # Append "modified_" to the object name
response = s3.put_object(Bucket=bucket_name, Key=object_name, Body=modified_image_bytes)

print(f"Modified image with prompt '{prompt}' uploaded to S3 bucket {bucket_name} with key {object_name}")