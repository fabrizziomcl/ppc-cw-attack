import numpy as np

def softmax(logits):
    """Calcula la función softmax para un vector de logits."""
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
    Calcula los componentes principales de una matriz de datos usando SVD.

    Args:
        row_vectors (np.ndarray): Matriz de (N_vectores, Dimension).

    Returns:
        tuple: (vector_medio, componentes_principales)
    """
    # 1. Calcular el vector medio y centrar los datos
    num_vectors = row_vectors.shape[0]
    mean_vector = np.mean(row_vectors, axis=0)
    X_centered = row_vectors - mean_vector  # forma (N, D)

    # 2. Descomposición SVD de la matriz centrada
    #    X_centered = U @ Sigma @ Vt
    U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 3. Los vectores fila de Vt son los componentes principales
    #    (cada fila de Vt es un eigenvector de la covarianza)
    principal_components = Vt.T  # forma (D, D), columnas = componentes

    return mean_vector, principal_components

def project_and_reconstruct(row_vectors, k, mean_vector, principal_components):
    """
    Proyecta y reconstruye los vectores fila usando una base de 'k' componentes.
    """
    subspace_base = principal_components[:, :k]  # (D, k)
    centered = row_vectors - mean_vector         # (N, D)
    projected = centered @ subspace_base         # (N, k)
    reconstructed = projected @ subspace_base.T  # (N, D)
    return reconstructed + mean_vector

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
        recon = project_and_reconstruct(row_vectors, k, mean_vector, principal_components)
        recon_img = recon.reshape(h, w, c)
        logits = fmodel.predictions(recon_img)

        if np.argmax(logits) == dominant_class:
            p = softmax(logits)[dominant_class]
            return k, p

    return None, None
