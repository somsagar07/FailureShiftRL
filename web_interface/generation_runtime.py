"""Inference-only Stable Diffusion precision and bounded prompt conditioning reuse."""
import copy
from collections import OrderedDict


def generation_precision(requested, device):
    if requested not in ('auto', 'float32'):
        raise ValueError('generation_precision must be auto or float32.')
    return 'float16' if requested == 'auto' and device == 'cuda' else 'float32'


class GenerationRuntime:
    """Own one immutable rendering pipeline and CPU-only conditioning tensors.

    The manager serializes inference and releases this runtime before Adapt.
    Training continues to use the separate, float32 load_diffusion path.
    """
    CACHE_BYTES = 8 * 1024 * 1024

    def __init__(self, pipeline, encoder, device, precision, *, cache_bytes=CACHE_BYTES):
        self.pipeline = pipeline
        self.torch = encoder.torch
        self.device = device
        self.precision = precision
        self.cache_limit_bytes = cache_bytes
        self._scope = None
        self._prompts = OrderedDict()
        self._negative = None
        self.cache_bytes = 0
        dtype = getattr(self.torch, precision)
        if pipeline.text_encoder is encoder.model and pipeline.text_encoder.dtype != dtype:
            # Analyze's feature definition is float32. Pipeline.to() traverses
            # its text encoder too, so detach it before changing any precision.
            pipeline.text_encoder = copy.deepcopy(pipeline.text_encoder)
        pipeline.to(device=device, dtype=dtype)

    @staticmethod
    def _bytes(tensor):
        return tensor.numel() * tensor.element_size()

    def _retain(self, prompt, tensor):
        size = self._bytes(tensor)
        negative_size = self._bytes(self._negative) if self._negative is not None else 0
        if size + negative_size > self.cache_limit_bytes:
            return
        while self._prompts and self.cache_bytes + size > self.cache_limit_bytes:
            _, previous = self._prompts.popitem(last=False)
            self.cache_bytes -= self._bytes(previous)
        self._prompts[prompt] = tensor.detach().to(device='cpu', copy=True)
        self.cache_bytes += size

    def conditioning(self, prompt):
        pipeline = self.pipeline
        guidance = getattr(pipeline.unet.config, 'time_cond_proj_dim', None) is None
        scope = (id(pipeline), id(pipeline.text_encoder), id(pipeline.tokenizer),
                 self.device, str(pipeline.text_encoder.device), pipeline.text_encoder.dtype,
                 str(pipeline.unet.device), pipeline.unet.dtype, guidance)
        if scope != self._scope:
            self._prompts.clear()
            self._negative = None
            self.cache_bytes = 0
            self._scope = scope
        positive = self._prompts.get(prompt)
        negative = self._negative
        if positive is not None:
            self._prompts.move_to_end(prompt)
            positive = positive.to(device=self.device)
        if negative is not None:
            negative = negative.to(device=self.device)
        if positive is not None and (negative is not None or not guidance):
            return positive, negative
        positive, negative = pipeline.encode_prompt(
            prompt if positive is None else None, device=self.device,
            num_images_per_prompt=1, do_classifier_free_guidance=guidance,
            prompt_embeds=positive, negative_prompt_embeds=negative)
        for value in (positive, negative):
            if value is not None and not self.torch.isfinite(value).all():
                raise ValueError('Generation returned nonfinite prompt conditioning; try generation_precision=float32.')
        if self._negative is None and negative is not None:
            size = self._bytes(negative)
            if size <= self.cache_limit_bytes:
                while self._prompts and self.cache_bytes + size > self.cache_limit_bytes:
                    _, previous = self._prompts.popitem(last=False)
                    self.cache_bytes -= self._bytes(previous)
                self._negative = negative.detach().to(device='cpu', copy=True)
                self.cache_bytes += size
        if prompt not in self._prompts:
            self._retain(prompt, positive)
        return positive, negative

    def render(self, prompt, *, steps, size, generator):
        import numpy as np
        positive, negative = self.conditioning(prompt)
        def validate_decoder_output(module, inputs, pixels):
            # Diffusers clamps decoder pixels before returning even numpy
            # output. Check the raw tensor before infinities become 0 or 1.
            if not self.torch.isfinite(pixels).all():
                raise ValueError('Generation returned nonfinite image pixels; try generation_precision=float32.')
        guard = self.pipeline.vae.decoder.register_forward_hook(validate_decoder_output)
        try:
            output = self.pipeline(prompt_embeds=positive, negative_prompt_embeds=negative,
                num_inference_steps=steps, height=size, width=size, generator=generator, output_type='np')
        finally:
            guard.remove()
        # PIL conversion would turn NaNs into ordinary black pixels. Validate
        # first, then use exactly the pipeline's normal conversion to PIL.
        if not np.isfinite(output.images).all():
            raise ValueError('Generation returned nonfinite image pixels; try generation_precision=float32.')
        output.images = self.pipeline.image_processor.numpy_to_pil(output.images)
        return output

    def describe(self):
        return {'generation_precision': self.precision,
                'conditioning_cache': {'max_bytes': self.cache_limit_bytes, 'storage': 'cpu'}}
