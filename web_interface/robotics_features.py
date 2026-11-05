"""Observe policy representations without extra policy decisions.

Numeric imports stay lazy: the robotics worker must bootstrap its runtime first.
The collector is optional, bounded, and never changes a model's inputs/outputs.
"""
from __future__ import annotations

import copy
import hashlib
import math


MAX_OBSERVATION_FEATURES = 2048
MAX_HIDDEN_FEATURES = 65536

# Exact code emitted by the bundled TimeConditionedPolicy TorchScript exports.
# These identify computation, not checkpoint weights or user-supplied metadata.
# A different exporter/implementation conservatively uses observation features.
_STARTER_CODE = {
    "forward": "c04c81764fafbbed854cb9e03ced69f6dbdab881e1c764e2e0ff71c63ca05f96",
    "logits": "b0b8057e7b25fce415aa20f5b8e01f3751ed649a1246ad330a0b3dc156659ef2",
    "network": "b2c66b54ca668f6e21bdbef875f862da04f7ca5ef2b25a4b166f5f6aa2baf70f",
    "Linear": "5ed428953f0a801293bf06e2220f31a45b152baa96d192187b459c8f259f7cb4",
    "Tanh": "2b3085d5f1d1c71946d76a51cb6263e42e5586ffd29c5f3d00907c8e100dae39",
}


def _known_starter(policy):
    import torch
    if not isinstance(policy, torch.jit.ScriptModule) or callable(getattr(policy, "select_action", None)):
        return False
    try:
        codes = [(policy.forward.code, "forward"), (policy.logits.code, "logits"),
                 (policy.network.code, "network")]
        children = list(policy.network.named_children())
        if [name for name, _ in children] != ["0", "1", "2", "3", "4"]:
            return False
        for (_, child), kind in zip(children, ("Linear", "Tanh", "Linear", "Tanh", "Linear")):
            if child.original_name != kind:
                return False
            codes.append((child.code, kind))
        if any(hashlib.sha256(code.encode()).hexdigest() != _STARTER_CODE[key] for code, key in codes):
            return False
        dimension = policy.mean.numel()
        if policy.mean.shape != (dimension,) or policy.std.shape != (dimension,):
            return False
        if policy.frequencies.shape != (4,) or not math.isfinite(policy.horizon) or policy.horizon <= 0:
            return False
        for index, shape in ((0, (128, dimension + 9)), (2, (128, 128)), (4, (4, 128))):
            layer = children[index][1]
            if layer.weight.shape != shape or layer.bias.shape != (shape[0],):
                return False
        return True
    except (AttributeError, RuntimeError, TypeError):
        return False


def _starter_hidden(policy, state):
    """Read the verified pure MLP prefix at the current, unadvanced clock.

    The original scripted action method still runs exactly once afterwards.
    Functional linear/tanh operations avoid invoking any mutable module method.
    """
    import torch
    normalized = ((state - policy.mean) / policy.std).clamp(-10., 10.)
    steps = torch.full((state.size(0),), float(policy.step), dtype=state.dtype, device=state.device)
    clock = steps.reshape(-1, 1) / policy.horizon
    phase = clock * policy.frequencies
    value = torch.cat((normalized, clock, torch.sin(phase), torch.cos(phase)), dim=1)
    for index in ("0", "2"):
        layer = getattr(policy.network, index)
        value = torch.tanh(torch.nn.functional.linear(value, layer.weight, layer.bias))
    return value


def _has_parameters(module):
    return module is not None and any(True for _ in module.parameters())


def _native_head(adapter):
    """Return only known native action-head inputs, never arbitrary last leaves."""
    import torch
    policy = adapter.policy
    fmt = adapter.metadata.get("format")
    if fmt == "sb3":
        native = getattr(policy, "policy", None)
        if native is None or not type(native).__module__.startswith(("stable_baselines3.", "sb3_contrib.")):
            return None
        if isinstance(getattr(native, "action_net", None), torch.nn.Module):
            actor = getattr(getattr(native, "mlp_extractor", None), "policy_net", None)
            learned = any(_has_parameters(m) for m in (actor, getattr(native, "features_extractor", None),
                                                       getattr(native, "lstm_actor", None)))
            if learned:
                return native.action_net, "policy.action_net.input", -1
        actor = getattr(native, "actor", None)
        mean = getattr(actor, "mu", None)
        if isinstance(mean, torch.nn.Module):
            if hasattr(actor, "latent_pi"):
                if _has_parameters(actor.latent_pi) or _has_parameters(actor.features_extractor):
                    return mean, "policy.actor.mu.input", -1
            elif isinstance(mean, torch.nn.Sequential):
                layers = list(mean.children())
                indices = [index for index, layer in enumerate(layers) if isinstance(layer, torch.nn.Linear)]
                if indices:
                    index = indices[-1]
                    learned = any(_has_parameters(m) for m in layers[:index]) or _has_parameters(actor.features_extractor)
                    if learned:
                        return layers[index], f"policy.actor.mu.{index}.input", -1
    elif fmt == "robomimic":
        native = getattr(policy, "policy", None)
        nets = getattr(native, "nets", {})
        actor = nets["policy"] if "policy" in nets else None
        if actor is not None and type(actor).__module__ == "robomimic.models.policy_nets":
            supported = {"ActorNetwork", "GaussianActorNetwork", "GMMActorNetwork", "RNNActorNetwork",
                         "RNNGMMActorNetwork", "TransformerActorNetwork", "TransformerGMMActorNetwork"}
            if type(actor).__name__ in supported and "decoder" in actor.nets:
                return actor.nets["decoder"], "policy.nets.policy.nets.decoder.input", -1
    elif fmt == "lerobot":
        # Python hooks inside a compiled model can trigger recompilation or be
        # bypassed. Preserve the configured inference engine and use observations.
        if getattr(getattr(policy, "config", None), "compile_model", False):
            return None
        module = type(policy).__module__
        if module == "lerobot.policies.act.modeling_act":
            return policy.model.action_head, "model.action_head.input", -1
        if module == "lerobot.policies.diffusion.modeling_diffusion":
            return policy.diffusion.unet.final_conv[-1], "diffusion.unet.final_conv.final.input", 1
        if module in {"lerobot.policies.smolvla.modeling_smolvla", "lerobot.policies.pi0.modeling_pi0",
                      "lerobot.policies.pi05.modeling_pi05"}:
            return policy.model.action_out_proj, "model.action_out_proj.input", -1
        if module == "lerobot.policies.pi0_fast.modeling_pi0_fast":
            return policy.model.paligemma_with_expert.paligemma.lm_head, "model.paligemma.lm_head.input", -1
    return None


def _observation_vector(value):
    """Bounded, deterministic samples of mapped numeric observations.

    Image pixels are sampled uniformly in flattened order; nothing random, learned,
    or action-derived is implied. Copy only the retained values off the device.
    """
    import numpy as np
    import torch
    items = sorted(value.items()) if isinstance(value, dict) else [("state", value)]
    arrays = []
    for key, item in items:
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.is_complex():
                continue
            array = item.detach()
        else:
            try:
                array = np.asarray(item)
            except (ValueError, TypeError):
                continue
            if array.dtype.kind not in "biuf":
                continue
        size = array.size if isinstance(array, np.ndarray) else array.numel()
        if size == 0:
            continue
        arrays.append((str(key), array))
    if not arrays or len(arrays) > MAX_OBSERVATION_FEATURES:
        raise ValueError("No bounded numeric observation features are available.")
    quota = MAX_OBSERVATION_FEATURES // len(arrays)
    schema, parts = [], []
    for key, array in arrays:
        schema.append((key, tuple(array.shape)))
        flat = array.reshape(-1)
        length = len(flat)
        if length > quota:
            indices = np.linspace(0, length - 1, quota, dtype=np.int64)
            if isinstance(flat, torch.Tensor):
                flat = flat.index_select(0, torch.as_tensor(indices, device=flat.device))
            else:
                flat = flat[indices]
        if isinstance(flat, torch.Tensor):
            flat = flat.float().cpu().numpy()
        parts.append(np.array(flat, dtype=np.float32, copy=True))
    return np.concatenate(parts), schema


class FeatureCapture:
    """Per-action candidate committed only after successful native inference."""
    def __init__(self, adapter):
        self.info = {"kind": "observation", "source_layer": "mapped_observation",
                     "dimension": None, "pooling": "uniform_flat_samples_per_numeric_input",
                     "max_dimensions": MAX_OBSERVATION_FEATURES,
                     "warning": "Hidden features are unavailable for this policy; using observation features."}
        self.mode = "observation"
        self.handles = []
        self.starter = None
        self.schema = None
        self.reset()
        self.bind(adapter)

    def _source(self, source, pooling):
        if self.info["dimension"] is not None and self.info["source_layer"] != source:
            raise ValueError("The policy feature source changed after its first observation.")
        self.info.update(kind="policy_hidden", source_layer=source, pooling=pooling,
                         max_dimensions=MAX_HIDDEN_FEATURES)
        self.info.pop("warning", None)

    def bind(self, adapter):
        self._detach()
        self.starter = None
        self.mode = "observation"
        try:
            if adapter.metadata.get("format") == "torchscript":
                self.mode = "script"
                if _known_starter(adapter.policy):
                    self.starter = adapter.policy
                    self._source("network.3", "mean_batch; verified_stateless_prefix_at_action_clock")
                return
            head = _native_head(adapter)
            if head is not None:
                module, source, axis = head
                self.handles.append(module.register_forward_pre_hook(self._hook(axis)))
                self._source(source, "mean_nonfeature_axes; last_forward_per_action")
                self.mode = "hook"
        except Exception as exc:
            # Capture availability must never prevent an otherwise supported policy
            # from loading. Do not silently switch an established feature space.
            self._detach()
            self._warning(exc)

    def _hook(self, axis):
        def observe(module, inputs):
            if self.active:
                self.capture(inputs[0], axis=axis)
            # Returning anything else can replace the module's input arguments.
            return None
        return observe

    def _warning(self, exc):
        self.candidate = None
        self.info["warning"] = f"Feature capture unavailable: {type(exc).__name__}: {exc}"[:512]

    def begin(self):
        self.candidate = self.pending = None
        self.active = True

    def input(self, value):
        try:
            if self.starter is not None:
                import torch
                with torch.inference_mode():
                    self.capture(_starter_hidden(self.starter, value))
            elif self.mode in ("observation", "script") and self.info["kind"] == "observation":
                vector, schema = _observation_vector(value)
                if self.schema is not None and schema != self.schema:
                    raise ValueError("Mapped observation names or shapes changed during feature capture.")
                self.schema = schema
                self.candidate = vector
        except Exception as exc:
            self._warning(exc)

    def script_output(self, features):
        """The same action call may return an empty tensor for a queue-only step."""
        try:
            import torch
            self._source("torchscript.output.features", "mean_nonfeature_axes; same_action_forward")
            self.candidate = None
            if isinstance(features, torch.Tensor) and features.numel() == 0:
                return
            self.capture(features)
        except Exception as exc:
            self._warning(exc)

    def capture(self, value, *, axis=-1):
        self.candidate = None
        try:
            import torch
            if not isinstance(value, torch.Tensor) or not value.numel() or value.ndim < 1:
                raise ValueError("The hidden representation must be a nonempty tensor.")
            dimension = value.shape[axis]
            if not 1 <= dimension <= MAX_HIDDEN_FEATURES:
                raise ValueError("The hidden representation exceeds the supported feature dimension.")
            with torch.inference_mode():
                # Pool before copying, keeping at most one small vector even for
                # many denoising iterations or large visual/token activations.
                vector = value.detach().float().movedim(axis, -1).reshape(-1, dimension).mean(0)
                self.candidate = vector.cpu().numpy().copy()
        except Exception as exc:
            self._warning(exc)

    def commit(self):
        import numpy as np
        self.active = False
        try:
            vector = self.candidate
            if vector is None:
                return
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError("Policy features must be a finite nonempty vector.")
            if self.info["dimension"] not in (None, len(vector)):
                raise ValueError("Policy feature dimension changed during inference.")
            self.info["dimension"] = len(vector)
            self.pending = vector.copy()
        except Exception as exc:
            self._warning(exc)
        finally:
            self.candidate = None

    def take(self):
        result, self.pending = self.pending, None
        return result

    def describe(self):
        return copy.deepcopy(self.info)

    def reset(self):
        self.active = False
        self.pending = self.candidate = None

    def _detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def close(self):
        self.reset()
        self._detach()
        self.starter = None
