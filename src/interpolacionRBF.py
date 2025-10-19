# src/interpolacionRBF.py
import numpy as np
from tkinter import messagebox

class InterpolacionRBF:
    def __init__(self):
        self.matriz_A = None
        self.vector_Y = None
        self.pesos = None

    # ------------------------------------------------------------
    # Calcular matriz de interpolación A y pesos W
    # ------------------------------------------------------------
    def calcular_pesos(self, funcion_activacion: np.ndarray, salidas: np.ndarray):
        """
        Calcula los pesos W resolviendo A * W = Y mediante pseudoinversa.
        """
        if funcion_activacion is None or salidas is None:
            messagebox.showerror("Error", "Faltan datos de activación o salidas.")
            return None

        # Agregar columna de 1's para el sesgo Wo
        n_patrones = funcion_activacion.shape[0]
        A = np.hstack((np.ones((n_patrones, 1)), funcion_activacion))
        Y = salidas.reshape(-1, 1)

        # Resolver W = (A^T * A)^-1 * A^T * Y  (usando pseudoinversa)
        W = np.linalg.pinv(A) @ Y

        self.matriz_A = A
        self.vector_Y = Y
        self.pesos = W

        return W

    # ------------------------------------------------------------
    # Generar texto de resultado
    # ------------------------------------------------------------
    def generar_resumen_texto(self):
        if self.matriz_A is None or self.vector_Y is None or self.pesos is None:
            return "Aún no se ha calculado la matriz de interpolación ni los pesos."

        texto = "=== MATRIZ DE INTERPOLACIÓN (A) ===\n"
        texto += np.array2string(self.matriz_A, precision=4, suppress_small=True)
        texto += "\n\n=== VECTOR DE SALIDAS (Y) ===\n"
        texto += np.array2string(self.vector_Y.T, precision=4, suppress_small=True)
        texto += "\n\n=== OPERACIÓN: A * W = Y  →  W = A⁺ * Y ===\n\n"
        texto += "=== PESOS CALCULADOS (W) ===\n"
        for i, w in enumerate(self.pesos.flatten()):
            texto += f"W{i} = {w:.4f}\n"
        return texto
