# Pega esto en attacks/carlini_wagner.py

from __future__ import division

import numpy as np
import logging

from .base import Attack
from .base import call_decorator
from utils import onehot_like

class AdamOptimizer:
    """Basic Adam optimizer implementation."""
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * (gradient**2)
        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)
        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

class CarliniWagnerL2Attack(Attack):
    """The L2 version of the Carlini & Wagner attack."""

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search_steps=9, max_iterations=1000,
                 confidence=0, learning_rate=5e-3,
                 initial_const=1e-3, abort_early=True):

        a = input_or_adv
        min_, max_ = a.bounds()

        # --- Transformaciones de espacio ---
        def to_attack_space(x):
            # Transformar x del espacio del modelo [min_, max_] al espacio de ataque (-inf, inf)
            # a través de tanh. Es la inversa de to_model_space.
            x = (x - min_) / (max_ - min_) # Rango [0, 1]
            x = x * 2 - 1 # Rango [-1, 1]
            x = np.arctanh(x * 0.999999) # Rango (-inf, inf)
            return x

        def to_model_space(w):
            # Transformar w del espacio de ataque (-inf, inf) al espacio del modelo [min_, max_]
            x = np.tanh(w)
            # Calcular el gradiente de esta transformación: d(x)/d(w)
            # Esto es crucial para la retropropagación
            grad = 1 - x**2
            x = (x + 1) / 2 # Rango [0, 1]
            x = x * (max_ - min_) + min_ # Rango [min_, max_]
            grad *= (max_ - min_) / 2
            return x, grad

        att_original = to_attack_space(a.original_image)
        const = initial_const
        lower_bound = 0.0
        upper_bound = 1e10

        for outer_step in range(binary_search_steps):
            print(f"\nBúsqueda binaria paso {outer_step+1}/{binary_search_steps}, const = {const:.1e}")
            att_perturbation = np.zeros_like(att_original)
            optimizer = AdamOptimizer(att_perturbation.shape)
            best_l2_for_const = float('inf')
            
            for i in range(max_iterations):
                w = att_original + att_perturbation
                x, dxdw = to_model_space(w) # x es la imagen, dxdw es el gradiente de la transformación

                logits, is_adv = a.predictions(x, strict=False)

                loss, dldx = self.loss_function(a, x, logits, confidence, const)

                # --- CORRECCIÓN CRÍTICA AQUÍ ---
                # Propagar el gradiente hacia atrás a través de la transformación
                # d(loss)/d(w) = d(loss)/d(x) * d(x)/d(w)
                # Esta es la regla de la cadena. dldx es d(loss)/d(x)
                gradient = dldx * dxdw

                att_perturbation += optimizer(gradient, learning_rate)

                if (i + 1) % 100 == 0:
                     print(f"  Iteración {i+1}/{max_iterations}, Pérdida: {loss:.4f}, Mejor L2: {best_l2_for_const:.4f}")

                if is_adv:
                    l2_dist = np.linalg.norm(x - a.original_image)
                    if l2_dist < best_l2_for_const:
                         best_l2_for_const = l2_dist

            # Actualizar la constante para la búsqueda binaria
            if best_l2_for_const < float('inf'):
                upper_bound = min(upper_bound, const)
                print(f"  Éxito para const={const:.1e}. Nuevo límite superior: {upper_bound:.1e}")
            else:
                lower_bound = max(lower_bound, const)
                print(f"  Fallo para const={const:.1e}. Nuevo límite inferior: {lower_bound:.1e}")

            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            else:
                const *= 10
        
        return

    @classmethod
    def loss_function(cls, a, x, logits, confidence, const):
        # Determinar las clases a comparar
        targeted = a.target_class() is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class())
            c_maximize = a.target_class()
        else:
            c_minimize = a.original_class
            c_maximize = cls.best_other_class(logits, a.original_class)
        
        # Pérdida de clasificación
        adv_loss_term = logits[c_minimize] - logits[c_maximize] + confidence
        loss_adv = np.maximum(0, adv_loss_term)
        
        # Pérdida de distancia L2
        loss_dist = np.sum((x - a.original_image)**2)

        # Pérdida total
        total_loss = loss_dist + const * loss_adv
        
        # --- Cálculo del Gradiente (d(loss)/d(x)) ---
        # Gradiente de la distancia
        grad_dist = 2 * (x - a.original_image)
        
        # Gradiente de la clasificación
        if adv_loss_term <= 0:
             grad_adv = 0
        else:
            grad_logits = np.zeros_like(logits)
            grad_logits[c_minimize] = 1
            grad_logits[c_maximize] = -1
            # a.backward() calcula d(loss_adv)/d(x)
            grad_adv = a.backward(grad_logits, x, strict=False) # <--- AÑADIMOS strict=False

        total_grad = grad_dist + const * grad_adv
        return total_loss, total_grad

    @staticmethod
    def best_other_class(logits, exclude):
        other_logits = logits - onehot_like(logits, exclude, value=np.inf)
        return np.argmax(other_logits)