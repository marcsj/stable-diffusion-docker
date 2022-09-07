#!/usr/bin/env python
import argparse, datetime, random, time
import torch
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


def iso_date_time():
    return datetime.datetime.now().isoformat()


def skip_safety_checker(images, *args, **kwargs):
    return images, False


def load_image(
    path
):
    image = Image.open(f"input/{path}").convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    return image


def load_pipeline(
    init_image, model_name, device, skip,
):
    print("load pipeline start:", iso_date_time())

    with open("token.txt") as f:
        token = f.read().replace("\n", "")

    if init_image is not None:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, revision="fp16", use_auth_token=token
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float32, revision="main", use_auth_token=token
        ).to(device)
    

    if skip:
        pipe.safety_checker = skip_safety_checker

    print("loaded pipeline:", iso_date_time())
    return pipe


def prompt_stable_diffusion(
    pipe, prompt, prefix, steps, scale, seed, samples, height, width, iters, device
):
    print("stable diffusion prompt: loaded models after:", iso_date_time())

    generator = torch.Generator(device=device).manual_seed(seed)
    for j in range(iters):
        with autocast(device):
            images = pipe(
                [prompt] * samples,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
            )

        for i, image in enumerate(images["sample"]):
            image.save(
                "output/%s__steps_%d__scale_%0.2f__seed_%d__n_%d.png"
                % (prefix, steps, scale, seed, j * samples + i + 1)
            )

    print("stable diffusion prompt: completed", iso_date_time(), flush=True)


def image_stable_diffusion(
    pipe, prompt, prefix, init_image, steps, scale, samples, iters, device, strength
):
    print("stable diffusion image: loaded models after:", iso_date_time())

    for j in range(iters):
        with autocast(device):
            images = pipe(
                prompt=prompt,
                init_image=init_image,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=scale,
            )

        for i, image in enumerate(images["sample"]):
            image.save(
                "output/%s_from_image_iter_%d.png"
                % (prefix, j * samples + i + 1)
            )

    print("stable diffusion image: completed", iso_date_time(), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Create images from a text prompt.")
    parser.add_argument(
        "prompt0",
        metavar="PROMPT",
        type=str,
        nargs="?",
        help="The prompt to render into an image",
    )
    parser.add_argument(
        "--prompt", type=str, nargs="?", help="The prompt to render into an image"
    )
    parser.add_argument(
        "--init_image", type=str, nargs="?", help="The image to use for image to image"
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="?",
        default=1,
        help="Number of images to create per run",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="?",
        default=1,
        help="Number of times to run pipeline",
    )
    parser.add_argument(
        "--height", type=int, nargs="?", default=512, help="Image height in pixels"
    )
    parser.add_argument(
        "--width", type=int, nargs="?", default=512, help="Image width in pixels"
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="?",
        default=7.5,
        help="Classifier free guidance scale",
    )
    parser.add_argument(
        "--strength",
        type=float,
        nargs="?",
        const=True,
        default=0.75,
        help="Strength of image to image diffusion",
    )
    parser.add_argument(
        "--seed", type=int, nargs="?", default=0, help="RNG seed for repeatability"
    )
    parser.add_argument(
        "--steps", type=int, nargs="?", default=50, help="Number of sampling steps"
    )
    parser.add_argument(
        "--half",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Use float16 (half-sized) tensors instead of float32",
    )
    parser.add_argument(
        "--skip",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip the safety checker",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        const=True,
        default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        const=True,
        default="cuda"
    )
    args = parser.parse_args()

    if args.prompt0 is not None:
        args.prompt = args.prompt0

    prefix = args.prompt.replace(" ", "_")[:170]

    if args.seed == 0:
        args.seed = torch.random.seed()

    # get pipeline for running stable diffusion
    pipe = load_pipeline(args.init_image, args.model, args.device, args.skip)

    # execute stable diffusion for each iteration per sample
    if args.init_image is not None:
        image = load_image(args.init_image)
        image_stable_diffusion(pipe, args.prompt, prefix, image, args.steps, args.scale, args.samples, args.iters, args.device, args.strength)
    else:
        prompt_stable_diffusion(pipe, args.prompt, prefix, args.steps, args.scale, args.seed, args.samples, args.height, args.width, args.iters, args.device)


if __name__ == "__main__":
    main()
