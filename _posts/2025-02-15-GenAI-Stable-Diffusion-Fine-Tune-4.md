---
list_title: Stable Diffusion | Fine-Tuning | Image Super Resolution
title:  Image Super Resolution
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

Another effective technique for fine-tuning a model is enhancing the resolution of the generated image. Unlike traditional image upscaling methods that rely on simple interpolation, image super-resolution leverages advanced algorithms, often powered by deep learning. These models learn high-frequency patterns and details from a dataset of high-resolution images, enabling them to produce superior-quality results when applied to low-resolution images.

### Img2img diffusion

As previously discussed, Stable Diffusion models do not rely solely on text for initial guidance; they can also use an image as a starting point. The idea here involves leveraging the text-to-image pipeline to generate a small image (e.g., 256x256) and then applying an image-to-image model to upscale it, achieving a higher resolution.

```python
# img2img pipeline
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    cache_dir = "/Volumes/ai-1t/diffuser"
).to("mps")

# upscale the image to 768 x 768
img2image_3x = img2img_pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    image = resized_raw_image, # the original low-res image
    strength = 0.3,
    number_of_inference_steps = 80,
    guidance_scale = 8,
    generator = torch.Generator("mps").manual_seed(3)
).images[0]
```

Below is a comparison between the raw image and the enhanced image. The model nearly enhanced every aspect of the image - from the eyebrows and eyelashes to the pupils and the mouth.

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2025/01/sd-upscale-base.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2025/01/sd-upscale-img2img.png"></div>
</div>
