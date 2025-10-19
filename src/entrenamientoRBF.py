# src/entrenamientoRBF.py
import numpy as np
from tkinter import messagebox

class EntrenamientoRBF:
    def __init__(self):
        self.centros_radiales = None
        self.error_optimo = None
        self.distancias = None
        self.funcion_activacion = None

    # ------------------------------------------------------------
    # Validar error de aproximación óptimo
    # ------------------------------------------------------------
    def set_error_optimo(self, valor: float):
        """Valida que el error esté entre 0 y 0.1."""
        if valor <= 0 or valor > 0.1:
            messagebox.showerror("Error inválido", "El error óptimo debe estar entre 0 y 0.1")
            return False
        self.error_optimo = valor
        return True

    # ------------------------------------------------------------
    # Calcular distancias euclidianas
    # ------------------------------------------------------------
    def calcular_distancias(self, X: np.ndarray, R: np.ndarray):
        """
        Calcula la distancia euclidiana entre cada vector de entrada (X_i)
        y cada centro radial (R_j).
        Devuelve una matriz D de tamaño (n_patrones, n_centros).
        """
        if X is None or R is None:
            messagebox.showerror("Datos faltantes", "Debe inicializar los centros radiales y cargar el dataset.")
            return None

        n_patrones, n_entradas = X.shape
        n_centros = R.shape[0]

        D = np.zeros((n_patrones, n_centros))
        for i in range(n_patrones):
            for j in range(n_centros):
                D[i, j] = np.sqrt(np.sum((X[i] - R[j]) ** 2))

        self.distancias = D
        return D

    # ------------------------------------------------------------
    # Calcular función de activación FA = D² * ln(D)
    # ------------------------------------------------------------
    def calcular_funcion_activacion(self):
        """Calcula la función FA = D² * ln(D) evitando log(0)."""
        if self.distancias is None:
            messagebox.showerror("Sin distancias", "Debe calcular las distancias primero.")
            return None

        D = self.distancias.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            FA = np.where(D > 0, D**2 * np.log(D), 0)

        self.funcion_activacion = FA
        return FA

    # ------------------------------------------------------------
    # Función auxiliar para generar resumen en texto
    # ------------------------------------------------------------
    def generar_resumen_texto(self):
        """Devuelve texto con los valores de D y FA."""
        if self.distancias is None or self.funcion_activacion is None:
            return "Aún no se han calculado distancias ni función de activación."

        texto = "=== DISTANCIAS EUCLIDIANAS (D) ===\n"
        texto += np.array2string(self.distancias, precision=4, suppress_small=True)
        texto += "\n\n=== FUNCIÓN DE ACTIVACIÓN (FA = D² * ln(D)) ===\n"
        texto += np.array2string(self.funcion_activacion, precision=4, suppress_small=True)
        return texto
