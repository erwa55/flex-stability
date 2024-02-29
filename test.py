import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

device = "cuda"
num_images_per_prompt = 2

# Loading the prior pipeline
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)

# Loading the decoder pipeline with additional parameters to ignore mismatched sizes
decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade",
    torch_dtype=torch.float16,
    ignore_mismatched_sizes=True,  # Add this line to ignore the mismatched sizes error
).to(device)

prompt = "Anthropomorphic cat dressed as a pilot"
negative_prompt = ""

# Generating image embeddings using the prior
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20
)

# Decoding the image embeddings into images
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.half(),  # Ensure embeddings are in the correct dtype
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images

# Now, decoder_output is a list with your PIL images
