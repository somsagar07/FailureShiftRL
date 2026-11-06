"""Hugging Face object detection and prediction-consistency analysis.

Consistency scores describe changes to the model's own baseline predictions; they
are not accuracy measurements and do not require or imply ground-truth labels.
Comparisons assume the two images share coordinates (photometric edits only).
"""
from pathlib import Path
import re


class Detector:
    """Run a pretrained object detector with lazy model dependency imports."""

    def __init__(self, model_name, device='auto', confidence_threshold=0.5):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        version = re.match(r'^(\d+)\.(\d+)', torch.__version__)
        if not version or tuple(map(int, version.groups())) < (2, 6):
            raise ValueError('Object detection requires PyTorch 2.6 or newer for safe weights-only loading.')
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError('Choose a Hugging Face model ID or local model directory.')
        if not 0 <= confidence_threshold <= 1:
            raise ValueError('The confidence threshold must be between 0 and 1.')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch = torch
        self.device = torch.device(device)
        self.confidence_threshold = float(confidence_threshold)
        local = Path(model_name).expanduser()
        name = str(local) if local.is_dir() else model_name
        options = {'trust_remote_code': False, 'local_files_only': local.is_dir()}
        self.processor = AutoImageProcessor.from_pretrained(name, **options)
        self.model = AutoModelForObjectDetection.from_pretrained(
            name, weights_only=True, **options).eval().to(self.device)

    def _inputs(self, images):
        return {key: value.to(self.device) for key, value in
                self.processor(images=images, return_tensors='pt').items()}

    def predict(self, image):
        image = image.convert('RGB')
        inputs = self._inputs(image)
        with self.torch.inference_mode():
            output = self.model(**inputs)
            sizes = self.torch.tensor([[image.height, image.width]], device=self.device)
            result = self.processor.post_process_object_detection(
                output, threshold=self.confidence_threshold, target_sizes=sizes)[0]
        labels = getattr(self.model.config, 'id2label', {}) or {}
        detections = []
        for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
            label_id = int(label.item())
            name = labels.get(label_id, labels.get(str(label_id), str(label_id)))
            detections.append({'label': str(name), 'score': float(score.item()),
                               'box': [float(value) for value in box.tolist()]})
        return {'detections': detections}

    def embeddings(self, images):
        """Return pooled backbone image features, one vector per input image.

        DETR's final convolutional map is pooled over unpadded positions. Models
        without a DETR-style backbone use their encoder output when available.
        """
        single = not isinstance(images, (list, tuple))
        batch = [images] if single else list(images)
        if not batch:
            raise ValueError('At least one image is required for embeddings.')
        inputs = self._inputs([image.convert('RGB') for image in batch])
        torch = self.torch
        with torch.inference_mode():
            backbone = getattr(getattr(self.model, 'model', None), 'backbone', None)
            if callable(backbone):
                pixels = inputs['pixel_values']
                mask = inputs.get('pixel_mask')
                if mask is None:
                    mask = torch.ones(pixels.shape[0], *pixels.shape[-2:], device=pixels.device, dtype=torch.bool)
                maps, _ = backbone(pixels, mask)
                features, valid = maps[-1]
                if valid is None:
                    pooled = features.mean(dim=(-2, -1))
                else:
                    valid = valid.to(dtype=features.dtype).unsqueeze(1)
                    pooled = (features * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)
            else:
                output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                features = getattr(output, 'encoder_last_hidden_state', None)
                if features is None:
                    raise ValueError('This detector does not expose compatible image backbone or encoder features.')
                pooled = features.mean(dim=1)
            if pooled.ndim != 2 or not torch.isfinite(pooled).all():
                raise ValueError('The detector returned invalid image features.')
            result = pooled.detach().to(device='cpu', dtype=torch.float32).numpy()
        return result[0] if single else result


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compare_predictions(baseline, transformed, iou_threshold=0.5):
    """Match same-class boxes and summarize changes to baseline predictions.

    Use only for edits preserving image geometry. Severity is the number of lost
    predictions plus positive matched confidence decreases, divided by the
    baseline count. Newly added predictions are counted but do not contribute
    to baseline-referenced severity. Neither result is a ground-truth claim.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError('The IoU threshold must be between 0 and 1.')
    before = baseline['detections']
    after = transformed['detections']
    candidates = []
    for old in before:
        eligible = [(index, _iou(old['box'], new['box'])) for index, new in enumerate(after)
                    if old['label'] == new['label']]
        candidates.append([index for index, overlap in sorted(eligible, key=lambda pair: -pair[1])
                           if overlap >= iou_threshold])
    # Augmenting paths avoid losing a possible match when boxes overlap multiple peers.
    assigned = {}
    def assign(old_index, seen):
        for new_index in candidates[old_index]:
            if new_index in seen:
                continue
            seen.add(new_index)
            if new_index not in assigned or assign(assigned[new_index], seen):
                assigned[new_index] = old_index
                return True
        return False
    for index in range(len(before)):
        assign(index, set())
    drops = [max(0.0, float(before[old]['score']) - float(after[new]['score']))
             for new, old in assigned.items()]
    matched = len(assigned)
    lost = len(before) - matched
    added = len(after) - matched
    return {'baseline_count': len(before), 'transformed_count': len(after),
            'matched_count': matched, 'dropped_count': lost, 'added_count': added,
            'mean_confidence_drop': sum(drops) / matched if matched else 0.0,
            'severity': (lost + sum(drops)) / len(before) if before else 0.0,
            'candidate': bool(lost or added or sum(drops) > 1e-8)}


def render_detections(image, prediction):
    """Draw prediction boxes on a copy of an image."""
    from PIL import ImageDraw

    result = image.convert('RGB').copy()
    draw = ImageDraw.Draw(result)
    for detection in prediction['detections']:
        box = detection['box']
        draw.rectangle(box, outline=(255, 50, 50), width=2)
        text = f"{detection['label']} {detection['score']:.2f}"
        draw.text((box[0], max(0, box[1] - 12)), text, fill=(255, 50, 50))
    return result
