from contextlib import ExitStack
from functools import partial
from diffusers import AutoPipelineForText2Image
import torch
from typing import Optional, Callable, Any, Literal
from PIL.Image import Image as PILImage



def load_sdxl_lightning(steps: int, device: str):
    '''Load SDXL-Lightning model with specified number of steps.'''

    # load dependencies
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # config
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    if steps == 1:
        ckpt = "sdxl_lightning_1step_unet_x0.safetensors"
    elif steps in [2,4,8]:
        ckpt = f"sdxl_lightning_{steps}step_unet.safetensors"
    else:
        raise ValueError(f"Invalid number of steps: {steps}")

    # Load model
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)

    # fix sampler
    extra_kwargs = dict(prediction_type="sample") if steps == 1 else {}
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", **extra_kwargs)

    return pipe



class SDResult:
    prompt : str
    seed : int
    representations : dict[str, list[torch.Tensor]]
    images : list[PILImage]
    result_latent : torch.Tensor
    result_tensor : torch.Tensor
    result_image : PILImage
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __repr__(self): return f'<SDResult prompt="{self.prompt}" seed={self.seed} ...>'


class SD:
    '''
    Usage:
    ```
    sd = SD('SD1.5')
    result = sd('a cat')
    result.result_image
    '''
    known_models = {
        'SD-1.5': {
            'name': 'runwayml/stable-diffusion-v1-5',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD-2.1': {
            'name': 'stabilityai/stable-diffusion-2-1',
            'steps': 50,
            'guidance_scale': 7.5,
        },
        'SD-Turbo': {
            'name': 'stabilityai/sd-turbo',
            'steps': 2,
            'guidance_scale': 0.0,
        },
        'SDXL-Turbo': {
            'name': 'stabilityai/sdxl-turbo',
            'steps': 4,
            'guidance_scale': 0.0,
        },
        **{f'SDXL-Lightning-{steps}step': {
            'name': f'ByteDance/SDXL-Lightning-{steps}step',
            'steps': steps,
            'guidance_scale': 0.0,
            'load_fn': partial(load_sdxl_lightning, steps),
        } for steps in [1,2,4,8]},
        'SDXL-Lightning': {  # default
            'name': 'ByteDance/SDXL-Lightning',
            'steps': 4,
            'guidance_scale': 0.0,
            'load_fn': partial(load_sdxl_lightning, 4),
        }
    }

    available_extract_positions = ['conv_in', 'down_blocks[0]', 'down_blocks[1]', 'down_blocks[2]', 'mid_block', 'up_blocks[0]', 'up_blocks[1]', 'up_blocks[2]', 'conv_out']

    def __init__(
            self,
            model_name: Literal['SD1.5', 'SD2.1', 'SD-Turbo', 'SDXL-Turbo'] | str = 'SD1.5',
            device: str = 'auto',
        ):
        self.model_name = model_name
        self.config = self.known_models.get(model_name, {'name':model_name})
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'

        # setup pipeline
        if 'load_fn' in self.config:
            self.pipeline = self.config['load_fn'](device=self.device)
        else:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.config['name'], torch_dtype=torch.float16).to(self.device)
        self.vae = self.pipeline.vae

        # upcast vae if necessary (SDXL models require float32)
        if hasattr(self.pipeline, 'upcast_vae') and self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.pipeline.upcast_vae()

        # check h-space dim
        # TODO

    @torch.no_grad()
    def vae_decode(self, latents):
        latents = latents.to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype) / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        return image

    def __call__(
            self,
            prompt: str,
            steps: Optional[int] = None,
            guidance_scale: Optional[float] = None,
            seed: Optional[int] = None,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
            modification: Optional[Callable[[Any,Any,Any,str],
            Optional[torch.Tensor]]] = None,
            preserve_grad: bool = False,
            extract_positions: list[str] = []
        ) -> SDResult:

        # use default values if not specified
        if steps is None:
            if 'steps' in self.config: steps = self.config['steps']
            else: raise ValueError('steps must be specified')
        if guidance_scale is None:
            if 'guidance_scale' in self.config: guidance_scale = self.config['guidance_scale']
            else: raise ValueError('guidance_scale must be specified')

        # random seed if not specified
        seed = seed if seed != None else int(torch.randint(0, 2**32, (1,)).item())

        # TODO: fix the following line for gradient preservation
        call_pipeline = (lambda *args, **kwargs: self.pipeline.__class__.__call__.__wrapped__(self.pipeline, *args, **kwargs)) if preserve_grad else self.pipeline

        # variables to store extracted results in
        representations = {pos: [] for pos in extract_positions}
        images = []

        def latents_callback(pipe, step_index, timestep, callback_kwargs):
            '''callback function to extract intermediate images'''
            latents = callback_kwargs['latents']
            image = (self.vae_decode(latents)[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            images.extend(self.pipeline.numpy_to_pil(image))
            return callback_kwargs

        # run pipeline
        with ExitStack() as stack, torch.no_grad():
            # setup hooks to extract representations
            for extract_position in self.available_extract_positions:
                def get_repr(module, input, output, extract_position):
                    if extract_position in extract_positions:
                        representations[extract_position].append(output)
                    if modification:
                        return modification(module, input, output, extract_position)
                # eval is unsafe. Do not use in production.
                stack.enter_context(eval(f'unet.{extract_position}', {'__builtins__': {}, 'unet': self.pipeline.unet}).register_forward_hook(partial(get_repr, extract_position=extract_position)))
            
            # run pipeline
            result = self.pipeline(
                prompt,
                width = width,
                height = height,
                num_inference_steps = steps,
                guidance_scale = guidance_scale,
                callback_on_step_end = latents_callback,
                callback_on_step_end_tensor_inputs = ['latents'],
                generator = torch.Generator(self.device).manual_seed(seed),
                output_type = 'latent',
            )

        # cast images to same dtype as vae
        result_tensor = self.vae_decode(result.images)
        result_image = self.pipeline.image_processor.postprocess(result_tensor.detach(), output_type='pil')

        # return results
        return SDResult(
            prompt=prompt,
            seed=seed,
            representations=representations,
            images=images,
            result_latent=result.images[0],
            result_tensor=result_tensor[0],
            result_image=result_image[0],
        )
