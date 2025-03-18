import os

import modal
from modal import App

current_file_directory = os.path.dirname(os.path.abspath(__file__))


def download_model_to_image():
    import transformers
    from huggingface_hub import snapshot_download

    MODEL_PATH = "black-forest-labs/FLUX.1-dev"

    snapshot_download(
        MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],
    )

    MODEL_PATH = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"

    snapshot_download(
        MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],
    )

    transformers.utils.move_cache()


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "libgl1-mesa-glx",
        "libglib2.0-dev",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "torch",
        "diffusers==0.30.2",
        "transformers",
        "accelerate",
        "sentencepiece",
        "opencv-python",
    )
    .pip_install("huggingface_hub")
    .pip_install("hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env(
        {
            "HF_TOKEN": "YOUR_HF_TOKEN",
        }
    )
    .run_function(download_model_to_image, timeout=86400)
)


app = App("inpaint-flux", image=image)


@app.function(
    timeout=86400,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    mounts=[modal.Mount.from_local_python_packages("fluxinpaint")],
)
def inpaint_flux(image, mask, prompt, neg_prompt="", num_inference_steps=15):
    # ok okoko okok
    import torch

    print(torch.cuda.is_available())
    # import sys
    # sys.path.append("/fluxinpaint")
    # import torch
    # from transformers.utils.hub import move_cache

    from diffusers.utils import check_min_version
    from fluxinpaint.controlnet_flux import FluxControlNetModel
    from fluxinpaint.pipeline_flux_controlnet_inpaint import (
        FluxControlNetInpaintingPipeline,
    )
    from fluxinpaint.transformer_flux import FluxTransformer2DModel

    check_min_version("0.30.2")
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dytpe=torch.bfloat16,
    )

    # Build pipeline
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)

    # Load image and mask

    generator = torch.Generator(device="cuda").manual_seed(24)

    size = image.size

    # Inpaint
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt=neg_prompt,
        true_guidance_scale=3.5,
    ).images[0]

    result.save("flux_inpaint.png")
    print("Successfully inpaint image")
    return result


@app.function()
def gsma_and_flux(im, num_inference_steps=5):
    import cv2
    import numpy as np
    import PIL

    inpaint_flux = modal.Function.lookup("inpaint-flux", "inpaint_flux")
    gsam = modal.Function.lookup("grounded-sam", "GroundedSam.run")

    im = im.resize((768, 768))
    gs_res = gsam.remote(im, "fish . hand . human")
    r = gs_res["results"]
    human_label_index = r[3].index("human")
    fish_mask_index = r[3].index("fish")
    human_mask = PIL.Image.fromarray(np.array(r[-2][human_label_index][0]))
    _ = PIL.Image.fromarray(np.array(r[-2][fish_mask_index][0]))

    hand_mask = human_mask.convert("RGB").resize((768, 768))

    mask_array = np.array(hand_mask)
    kernel = np.ones((5, 5), np.uint8)  # Adjust size as needed
    eroded_mask = cv2.dilate(mask_array, kernel, iterations=10)
    hand_mask = PIL.Image.fromarray(eroded_mask)

    filled_image = inpaint_flux.remote(
        im,
        hand_mask,
        prompt="fish body",
        neg_prompt="hand fingers human, fish-eye",
        num_inference_steps=num_inference_steps,
    )

    overlay_color = (255, 0, 0)  # Red

    colored_mask = PIL.Image.new("RGB", filled_image.size, overlay_color)
    colored_mask = PIL.Image.composite(
        colored_mask, filled_image, hand_mask.convert("L")
    )

    overlayed_image = PIL.Image.blend(
        filled_image, colored_mask, alpha=0.5
    )  # Adjust alpha for transparency
    gs_res_after_fill = gsam.remote(filled_image, "fish . hand . human")

    res_after_fill_and_inpaint = PIL.Image.fromarray(gs_res_after_fill["results"][0])

    return (
        im,
        overlayed_image,
        gs_res_after_fill,
        res_after_fill_and_inpaint,
        filled_image,
    )
