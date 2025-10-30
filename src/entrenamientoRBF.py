# src/entrenamientoRBF.py
import numpy as np
from tkinter import messagebox

class EntrenamientoRBF:
    def __init__(self):
        self.centros_radiales = None
        self.error_optimo = None
        self.distancias = None
        self.funcion_activacion = None
        self.patrones = None  # guardamos los X usados (por fila) para mostrar

    def set_error_optimo(self, valor: float):
        if valor <= 0 or valor > 0.1:
            messagebox.showerror("Error inválido", "El error óptimo debe estar entre 0 y 0.1")
            return False
        self.error_optimo = valor
        return True

    def calcular_distancias(self, X: np.ndarray, R: np.ndarray):
        if X is None or R is None:
            messagebox.showerror("Datos faltantes", "Debe inicializar los centros radiales y cargar el dataset.")
            return None

        # Guardar patrones (orden y valores exactos)
        self.patrones = np.asarray(X, dtype=float)

        # Vectorized distance: broadcasting and norm over last axis
        # Shape: (N, 1, n_inputs) - (1, M, n_inputs) -> (N, M, n_inputs)
        diff = self.patrones[:, None, :] - np.asarray(R)[None, :, :]
        D = np.linalg.norm(diff, axis=2)  # (N, M)

        # Guardar con suficiente precisión
        self.distancias = np.round(D, 6)
        return self.distancias

    def calcular_funcion_activacion(self):
        if self.distancias is None:
            messagebox.showerror("Sin distancias", "Debe calcular las distancias primero.")
            return None

        D = self.distancias.copy()
        D_safe = np.where(D <= 0, 1e-12, D)
        with np.errstate(divide='ignore', invalid='ignore'):
            FA = D_safe**2 * np.log(D_safe)
        FA = np.nan_to_num(FA, nan=0.0, posinf=0.0, neginf=0.0)
        self.funcion_activacion = np.round(FA, 6)
        return self.funcion_activacion

    def generar_resumen_texto(self, max_rows: int = 20):
        if self.distancias is None or self.funcion_activacion is None or self.patrones is None:
            return "Aún no se han calculado distancias ni función de activación."

        N, M = self.distancias.shape
        lines = []
        lines.append("=== MAPEADO PATRÓN -> DISTANCIAS (D) ===\n")
        for i in range(min(N, max_rows)):
            xvec = np.array2string(self.patrones[i], precision=4, separator=", ")
            drow = np.array2string(self.distancias[i], precision=4, separator=", ")
            farow = np.array2string(self.funcion_activacion[i], precision=4, separator=", ")
            lines.append(f"Patrón {i}: X={xvec}\n  D: {drow}\n  FA: {farow}\n")
        if N > max_rows:
            lines.append(f"... (se mostraron {max_rows} de {N} patrones)\n")
        return "\n".join(lines)
