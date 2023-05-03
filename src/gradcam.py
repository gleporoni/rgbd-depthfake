import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        self.model.eval()
        
        self.activations_hook = self.target_layer.register_forward_hook(self.save_feature_maps)
        self.gradients_hook = self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x):
        device = next(self.model.parameters()).device
        
        # Forward pass
        logits = self.model(x.to(device))
        logits.backward()
        
        # Compute weights
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        weights = F.relu(weights)
        
        # Compute Grad-CAM
        grad_cam = torch.mul(self.feature_maps, weights)
        grad_cam = torch.sum(grad_cam, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        grad_cam = grad_cam.cpu().numpy()
        
        return grad_cam[0, 0]
    
    def remove_hooks(self):
        self.activations_hook.remove()
        self.gradients_hook.remove()
