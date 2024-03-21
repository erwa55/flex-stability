import boto3
from PIL import Image
import io

# Your AWS and model setup as before
s3 = boto3.client('s3')
bucket_name = 'flex-saas-demo-demo-temp'
image_key = 'your-object-name.jpg'

# Download image from S3
s3_response_object = s3.get_object(Bucket=bucket_name, Key=image_key)
image_content = s3_response_object['Body'].read()
image = Image.open(io.BytesIO(image_content))

# Assuming you have a function to send the image and prompt to the model
# This is highly dependent on the API or library you're using
def modify_image_with_model(image, prompt):

    # Pseudo-code for sending the image and prompt to the model
    # modified_image = model.modify_image(image=image, prompt=prompt)
    # return modified_image

    # Placeholder return
    return image  # This should be replaced with the actual modified image

# Modify the image with a specific prompt
prompt = "Put a rainbow background"
modified_image = modify_image_with_model(image, prompt)

# Save the modified image to a buffer
buffer = io.BytesIO()
modified_image.save(buffer, format="JPEG")
buffer.seek(0)

# Upload the modified image back to S3
modified_image_key = "modified_image_with_rainbow.jpg"
s3.upload_fileobj(buffer, bucket_name, modified_image_key)

