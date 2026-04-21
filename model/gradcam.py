import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def apply_gradcam_overlay(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img_array = np.array(original_image.convert("RGB"))
    cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_array)
    return overlay


def get_gradcam_b64(model, input_tensor: torch.Tensor, original_image: Image.Image, class_idx: int = None) -> str:
    import base64
    import io

    target_layer = model.features[-1]
    gcam = GradCAM(model, target_layer)

    try:
        cam = gcam.generate(input_tensor, class_idx)
        overlay = apply_gradcam_overlay(original_image, cam)
        pil_overlay = Image.fromarray(overlay)
        buffer = io.BytesIO()
        pil_overlay.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    finally:
        gcam.remove_hooks()