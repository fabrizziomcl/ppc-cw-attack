# Contenido completo de pca_detector.py (Versión Ultra-Manual)
import numpy as np

# --- Funciones de Utilidad Matemática ---

def softmax(logits):
    """Calcula la función softmax para un vector de logits."""
    # Desplazar los logits para evitar overflow/underflow en la exponencial
    stable_logits = logits - np.max(logits)
    exponentials = np.exp(stable_logits)
    return exponentials / np.sum(exponentials)

# --- Funciones Principales de Detección ---

def get_row_vectors(image_np):
    """Convierte una imagen (H, W, C) en una matriz de vectores fila (H, W*C)."""
    h, w, c = image_np.shape
    return image_np.reshape(h, w * c)

def compute_principal_components(row_vectors):
    """
    Calcula los componentes principales de una matriz de datos.
    Implementación manual de la covarianza y ordenación.

    Args:
        row_vectors (np.ndarray): Matriz de (N_vectores, Dimension).

    Returns:
        tuple: (vector_medio, componentes_principales)
    """
    num_vectors = row_vectors.shape[0]

    # 1. Calcular el vector medio y centrar los datos
    mean_vector = np.sum(row_vectors, axis=0) / num_vectors
    centered_vectors = row_vectors - mean_vector
    
    # 2. Calcular la matriz de covarianza manualmente.
    # C = (1 / (N-1)) * X_centered^T @ X_centered
    # (N-1 es para la covarianza muestral insesgada)
    # X_centered^T tiene forma (Dim, N), X_centered tiene forma (N, Dim)
    # El resultado es una matriz (Dim, Dim)
    covariance_matrix = (centered_vectors.T @ centered_vectors) / (num_vectors - 1)
    
    # 3. Calcular eigenvectores y eigenvalues de la matriz de covarianza.
    # Esta es la única función de alto nivel que es indispensable.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 4. Ordenar los componentes (eigenvectores) por su eigenvalue descendente.
    # Implementación manual del ordenamiento.
    # Creamos tuplas de (eigenvalue, eigenvector)
    eigen_pairs = []
    for i in range(len(eigenvalues)):
        # Guardamos el eigenvector como un vector columna
        eigen_pairs.append((eigenvalues[i], eigenvectors[:, i].reshape(-1, 1)))

    # Ordenar las tuplas por el eigenvalue (el primer elemento) en orden descendente
    eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)

    # Reconstruir la matriz de componentes principales ordenada
    # np.hstack apila los vectores columna horizontalmente
    principal_components = np.hstack([pair[1] for pair in eigen_pairs])
    
    return mean_vector, principal_components

def project_and_reconstruct(row_vectors, k, mean_vector, principal_components):
    """
    Proyecta y reconstruye los vectores fila usando una base de 'k' componentes.
    """
    # La base del subespacio son los primeros 'k' componentes principales
    subspace_base = principal_components[:, :k]
    
    # Centrar los datos
    centered_vectors = row_vectors - mean_vector
    
    # Proyección de datos en la base del subespacio para obtener coordenadas
    # Coords = X_centered @ U_k
    projected_coords = centered_vectors @ subspace_base
    
    # Reconstrucción desde las coordenadas de vuelta al espacio original
    # X_reconstructed_centered = Coords @ U_k^T
    reconstructed_centered = projected_coords @ subspace_base.T
    
    # Deshacer el centrado
    reconstructed_vectors = reconstructed_centered + mean_vector
    return reconstructed_vectors

def find_kp_point(image_np, fmodel):
    """
    Calcula el punto (k, p) para una imagen dada.
    """
    h, w, c = image_np.shape
    
    original_logits = fmodel.predictions(image_np)
    dominant_class = np.argmax(original_logits)
    
    row_vectors = get_row_vectors(image_np)
    try:
        mean_vector, principal_components = compute_principal_components(row_vectors)
    except np.linalg.LinAlgError:
        return None, None
        
    max_k = row_vectors.shape[0]
    
    for k in range(1, max_k + 1):
        reconstructed_vectors = project_and_reconstruct(row_vectors, k, mean_vector, principal_components)
        reconstructed_image_np = reconstructed_vectors.reshape(h, w, c)
        reconstructed_logits = fmodel.predictions(reconstructed_image_np)
        
        if np.argmax(reconstructed_logits) == dominant_class:
            p = softmax(reconstructed_logits)[dominant_class]
            return k, p
            
    return None, None