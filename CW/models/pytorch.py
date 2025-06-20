# Pega esto en models/pytorch.py (VERSIÓN FINAL CON CORRECCIÓN DE DTYPE)

from __future__ import absolute_import
import numpy as np
import warnings
from .base_models import DifferentiableModel

class PyTorchModel(DifferentiableModel):
    def __init__(
            self, model, bounds, num_classes,
            channel_axis=1, device=None, preprocessing=(0, 1)):
        import torch
        super(PyTorchModel, self).__init__(bounds=bounds,
                                           channel_axis=channel_axis,
                                           preprocessing=preprocessing)
        self._num_classes = num_classes
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self._model = model.to(self.device)
        if hasattr(model, 'training') and model.training:
            warnings.warn('The PyTorch model is in training mode. Call eval() for deterministic behavior.')

    def _transpose(self, x):
        if self.channel_axis() == 1:
            if x.ndim == 3: return np.transpose(x, (2, 0, 1))
            elif x.ndim == 4: return np.transpose(x, (0, 3, 1, 2))
        return x

    def _transpose_gradient(self, grad):
        if self.channel_axis() == 1:
            if grad.ndim == 3: return np.transpose(grad, (1, 2, 0))
            elif grad.ndim == 4: return np.transpose(grad, (0, 2, 3, 1))
        return grad
        
    def batch_predictions(self, images):
        import torch
        images, _ = self._process_input(images)
        images = self._transpose(images)
        images = torch.from_numpy(images.copy()).to(self.device).float() # <--- CORRECCIÓN AQUÍ

        # Quitar el print de DEBUG
        # print(f"DEBUG: La forma de la imagen antes de entrar al modelo es: {images.shape}")
        
        with torch.no_grad():
            predictions = self._model(images)
        predictions = predictions.to("cpu").detach().numpy()
        return predictions

    def num_classes(self):
        return self._num_classes

    def predictions_and_gradient(self, image, label):
        import torch
        import torch.nn as nn
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        image = self._transpose(image)
        
        target = torch.tensor([label], device=self.device)
        images = torch.from_numpy(image[np.newaxis].copy()).to(self.device).float() # <--- CORRECCIÓN AQUÍ
        images.requires_grad_()

        predictions = self._model(images)
        loss = nn.CrossEntropyLoss()(predictions, target)
        loss.backward()
        
        grad = images.grad.to("cpu").detach().numpy().squeeze(axis=0)
        grad = self._transpose_gradient(grad)
        grad = self._process_gradient(dpdx, grad)
        
        predictions = predictions.to("cpu").detach().numpy().squeeze(axis=0)
        return predictions, grad

    def backward(self, gradient, image, strict=True):
        import torch 
        
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        image = self._transpose(image)
        
        gradient_tensor = torch.from_numpy(gradient[np.newaxis, :]).to(self.device).float() # <--- AÑADIMOS la dimensión de lote
        images = torch.from_numpy(image[np.newaxis].copy()).to(self.device).float() # <--- CORRECCIÓN AQUÍ
        images.requires_grad_()
        
        predictions = self._model(images)
        predictions.backward(gradient=gradient_tensor)
        
        grad = images.grad.to("cpu").detach().numpy().squeeze(axis=0)
        grad = self._transpose_gradient(grad)
        grad = self._process_gradient(dpdx, grad)
        return grad