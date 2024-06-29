

import io
from pathlib import Path
from pydantic import BaseModel
from modal import (
    App,
    Image,
    Mount,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
)
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

sdxl_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.26.3",
        "invisible_watermark==0.2.0",
        "transformers~=4.38.2",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
    )
)

class InferenceRequest(BaseModel):
    prompt: str
    n_steps: int = 24
    high_noise_frac: float = 0.8

app = App("stable-diffusion-xl")

with sdxl_image.imports():
    import torch
    from diffusers import DiffusionPipeline

@app.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=sdxl_image)
class Model:
    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
        )
        snapshot_download(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            ignore_patterns=ignore,
        )

    @enter()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        self.base.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_stream.seek(0)
        
        chunk_size = 1024
        while True:
            chunk = byte_stream.read(chunk_size)
            if not chunk:
                break
            yield chunk

    @method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return b''.join(self._inference(prompt, n_steps=n_steps, high_noise_frac=high_noise_frac))

    @web_endpoint(method='POST', docs=True)
    async def web_inference(self, request: InferenceRequest):
        return StreamingResponse(
            self._inference(
                request.prompt, n_steps=request.n_steps, high_noise_frac=request.high_noise_frac
            ),
            media_type="image/jpeg",
        )

@app.local_entrypoint()
def main(prompt: str = "Unicorns and leprechauns sign a peace treaty"):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)

@app.function(
    allow_concurrent_inputs=20,
)
@asgi_app()
def backend():
    web_app = FastAPI()

    @web_app.post("/")
    async def read_root(request: Request):
        data = await request.json()
        print(data)
        inference_request = InferenceRequest(**data)
        model = Model()
        return StreamingResponse(
            model._inference(
                inference_request.prompt,
                n_steps=inference_request.n_steps,
                high_noise_frac=inference_request.high_noise_frac
            ),
            media_type="image/jpeg"
        )

    return web_app