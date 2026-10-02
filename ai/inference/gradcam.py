"""
ai/inference/gradcam.py
--------------------------
Grad-CAM (Gradient-weighted Class Activation Mapping) for the CNN.

WHY IT MATTERS (explain this in your interview):
Deep CNNs are "black boxes" - they give a prediction but not a reason.
Grad-CAM answers "WHERE in the image did the model look to make this
decision?" It does this by:
  1. Running a forward pass and picking the target class's score.
  2. Backpropagating that score's gradient only as far as the LAST
     convolutional layer (not all the way to the input pixels).
  3. Using those gradients to weight each of that layer's feature-map
     channels - channels that mattered more for the prediction get a
     bigger weight.
  4. Summing the weighted channels into a single heatmap, then upsampling
     it to the input image's size.

This produces a heatmap that lights up the pixels the network actually
relied on - useful for debugging (is it looking at the person, or just the
background?) and for building trust in a safety/security product.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, 3, 224, 224)
        Returns a normalized heatmap (224, 224) in range [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1))

        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> one weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_bgr_uint8, heatmap_0_1, alpha=0.4):
    """Blend a Grad-CAM heatmap onto the original image for display."""
    heatmap_color = cv2.applyColorMap((heatmap_0_1 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
