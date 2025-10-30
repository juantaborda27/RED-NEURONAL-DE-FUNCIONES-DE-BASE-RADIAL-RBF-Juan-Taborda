# guardar_resultados.py
import json
from tkinter import messagebox, filedialog
import numpy as np
from pathlib import Path

class GuardarResultadosRBF:
    def __init__(self, carpeta_salida="resultados"):
        self.carpeta = Path(carpeta_salida)
        self.carpeta.mkdir(exist_ok=True)

    def guardar(self, resumen, centros, distancias, fa, matriz_interp, pesos):
        if resumen is None:
            messagebox.showwarning("Sin datos", "No hay resumen de entrenamiento para guardar.")
            return

        # Pedir nombre del archivo al usuario
        ruta_archivo = filedialog.asksaveasfilename(
            title="Guardar entrenamiento RBF como...",
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json")],
            initialdir=str(self.carpeta),
            initialfile="entrenamiento_rbf.json"
        )

        if not ruta_archivo:  # si el usuario cancela
            messagebox.showinfo("Cancelado", "Guardado cancelado por el usuario.")
            return

        # Determinar nombres de columnas de entrada, si están en resumen
        input_names = None
        if isinstance(resumen, dict):
            cols = resumen.get("columns") or resumen.get("columns_after") or None
            if cols:
                # asumimos que la última columna es Y
                input_names = cols[:-1] if len(cols) > 1 else cols

        # Convertir todos los arrays a listas normales
        data = {
            "resumen": resumen,
            # guardar las columnas originales también (siempre útil)
            "columns": resumen.get("columns") if isinstance(resumen, dict) and resumen.get("columns") else None,
            # lista explícita de nombres de entradas (para la simulación)
            "input_names": input_names if input_names is not None else None,
            "centros_radiales": centros.tolist() if centros is not None else None,
            "distancias": distancias.tolist() if distancias is not None else None,
            "funcion_activacion": fa.tolist() if fa is not None else None,
            "matriz_interpolacion": matriz_interp.tolist() if matriz_interp is not None else None,
            "pesos": {f"W{i}": float(w) for i, w in enumerate(pesos)} if pesos is not None else None
        }

        try:
            with open(ruta_archivo, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Guardado exitoso", f"Entrenamiento guardado como:\n{ruta_archivo}")
        except Exception as e:
            messagebox.showerror("Error al guardar", f"No se pudo guardar el archivo JSON:\n{e}")
