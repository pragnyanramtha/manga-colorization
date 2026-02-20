"""Core manga colorization pipeline using SD1.5 + ControlNet + IP-Adapter."""

import torch
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPVisionModelWithProjection


class MangaColorizerPipeline:
    """Reference-based manga colorization using Stable Diffusion 1.5.

    Uses ControlNet for structural guidance (lineart) and IP-Adapter
    for color/style transfer from a reference image.
    """

    NEGATIVE_PROMPT = "monochrome, greyscale, lowres, bad anatomy"

    # Default model identifiers
    DEFAULT_CHECKPOINT_URL = "https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors"
    DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11p_sd15s2_lineart_anime"
    DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
    DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter-plus_sd15.safetensors"
    DEFAULT_SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def __init__(
        self,
        checkpoint_url: str | None = None,
        controlnet_id: str | None = None,
        ip_adapter_repo: str | None = None,
    ):
        checkpoint_url = checkpoint_url or self.DEFAULT_CHECKPOINT_URL
        controlnet_id = controlnet_id or self.DEFAULT_CONTROLNET_ID
        ip_adapter_repo = ip_adapter_repo or self.DEFAULT_IP_ADAPTER_REPO

        # Load ViT-H image encoder (required for IP-Adapter Plus)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_repo,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16,
        )

        # Load text encoder with skip-last-layer (recommended for s2 ControlNet)
        text_encoder = CLIPTextModel.from_pretrained(
            self.DEFAULT_SD15_REPO,
            subfolder="text_encoder",
            num_hidden_layers=11,
            torch_dtype=torch.float16,
        )

        # Build pipeline from single-file checkpoint
        self.pipe = StableDiffusionControlNetPipeline.from_single_file(
            checkpoint_url,
            controlnet=controlnet,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # Load IP-Adapter (MUST happen before enable_model_cpu_offload)
        self.pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder="models",
            weight_name=self.DEFAULT_IP_ADAPTER_WEIGHT,
        )
        self.pipe.set_ip_adapter_scale(0.6)

        # Scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Memory optimization (MUST be after load_ip_adapter)
        self.pipe.enable_model_cpu_offload()

    def _resize_for_sd(self, image: Image.Image) -> Image.Image:
        """Resize image to SD1.5 native resolution.

        Portrait/square -> 512x768, landscape -> 768x512.
        """
        w, h = image.size
        if w > h:
            target = (768, 512)
        else:
            target = (512, 768)
        return image.resize(target, Image.LANCZOS)

    def colorize_panel(
        self,
        bw_image_path: str,
        reference_image_path: str,
        character_lora: str | None = None,
        num_inference_steps: int = 20,
        controlnet_scale: float = 1.1,
        ip_adapter_scale: float = 0.6,
        seed: int | None = None,
    ) -> Image.Image:
        """Colorize a B&W manga panel using a colored reference image.

        Args:
            bw_image_path: Path to the B&W manga panel
            reference_image_path: Path to the colored reference/character sheet
            character_lora: Optional path to a LoRA .safetensors file
            num_inference_steps: Number of diffusion steps (default 20)
            controlnet_scale: ControlNet conditioning strength (default 1.1)
            ip_adapter_scale: IP-Adapter influence strength (default 0.6)
            seed: Random seed for reproducibility

        Returns:
            PIL Image of the colorized panel
        """
        bw_image = Image.open(bw_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        bw_image = self._resize_for_sd(bw_image)
        reference_image = self._resize_for_sd(reference_image)

        # Update IP-Adapter scale if changed
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Load optional LoRA
        if character_lora:
            self.pipe.load_lora_weights(character_lora)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipe(
            prompt="high quality, anime, detailed, vibrant colors",
            negative_prompt=self.NEGATIVE_PROMPT,
            image=bw_image,
            ip_adapter_image=reference_image,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Unload LoRA if loaded
        if character_lora:
            self.pipe.unload_lora_weights()

        return result.images[0]
