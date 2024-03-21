import DiffusionPipeline
import torch
import boto3
from PIL import Image
from io import BytesIO

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Download the image from S3
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'  # Replace with your bucket name
object_name = 'your-object-name.jpg'  # Replace with the object name of the image to modify

# Note: This step downloads the image but doesn't use it in the modification process,
# as the model generates a new image from the prompt.
response = s3.get_object(Bucket=bucket_name, Key=object_name)
image_content = response['Body'].read()
# If you wanted to display or directly manipulate the downloaded image, you could do so like this:
# existing_image = Image.open(BytesIO(image_content))

# Generate a new image based on the prompt (this is not a modification of the existing image)
prompt = "Add a rainbow "
images = pipe(prompt=prompt).images[0]

# Convert the PIL image to bytes
image_byte_array = BytesIO()
images.save(image_byte_array, format='JPEG')  # You can change 'JPEG' to 'PNG' if you prefer
image_bytes = image_byte_array.getvalue()

# Optionally, you can overwrite the existing object or specify a new name
new_object_name = 'modified-your-object-name.jpg'  # Replace with your desired object name for the new image

# Upload the new image to S3
response = s3.put_object(Bucket=bucket_name, Key=new_object_name, Body=image_bytes)

print(f"New image uploaded to S3 bucket {bucket_name} with key {new_object_name}")
