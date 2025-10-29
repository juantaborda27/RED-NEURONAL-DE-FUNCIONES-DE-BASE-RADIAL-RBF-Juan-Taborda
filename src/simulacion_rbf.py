# simulacion_rbf.py
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

def _pesos_from_json(obj):
    """Normaliza los pesos guardados en JSON a un array 1D (intenta ordenar por clave si es dict)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return np.array([v for _, v in items], dtype=float)
    return np.array(obj, dtype=float)

def _inferir_sigma_robusto(centros, distancias=None, fa=None):
    """(Queda por compatibilidad; no usado en este flujo principal con FA= D^2 * ln(D))."""
    try:
        if distancias is not None and fa is not None:
            D = np.asarray(distancias, dtype=float)
            F = np.asarray(fa, dtype=float)
            mask = (F > 0) & (F < 1) & (D > 0)
            if np.any(mask):
                vals = - (D[mask]**2) / (2.0 * np.log(F[mask]))
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size > 0:
                    sigma_est = float(np.median(np.sqrt(vals)))
                    if sigma_est > 0 and np.isfinite(sigma_est):
                        return sigma_est
    except Exception:
        pass
    try:
        C = np.asarray(centros, dtype=float)
        if C.ndim == 2 and C.shape[0] > 1:
            D2 = np.sum((C[:, None, :] - C[None, :, :])**2, axis=2)
            np.fill_diagonal(D2, np.inf)
            D = np.sqrt(D2)
            nn = np.min(D, axis=1)
            nn = nn[np.isfinite(nn)]
            if nn.size > 0:
                d_med = float(np.median(nn))
                if d_med > 0:
                    return float(d_med / np.sqrt(2.0 * np.log(2.0)))
    except Exception:
        pass
    return 1.0

class SimulacionPanel:
    """
    Panel de simulación centrado en usar los datos guardados en el JSON.
    """
    def __init__(self, parent_container, app_ref=None):
        self.parent = parent_container
        self.app = app_ref

        # limpiar container y montar panel
        for w in self.parent.winfo_children():
            w.destroy()

        self.frame = ttk.Frame(self.parent, padding=8)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # estado
        self.model_json = None
        self.centros = None
        self.distancias = None
        self.fa_train = None
        self.matriz_A = None
        self.pesos = None
        self.resumen = None
        self.sigma = None

        self.n_inputs = 0
        self.input_names = []
        self.input_entries = []
        self.manual_patterns = []  # lista de np.array

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.frame)
        top.pack(fill=tk.X, pady=4)

        ttk.Button(top, text="Cargar JSON (modelo entrenado)", command=self._cargar_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Agregar patrón desde entradas", command=self._agregar_patron_manual_from_entries).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Simular patrones manuales", command=lambda: self._ejecutar_simulacion(manuales_only=True)).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Limpiar patrones manuales", command=self._limpiar_manuales).pack(side=tk.LEFT, padx=4)

        area = ttk.Frame(self.frame)
        area.pack(fill=tk.BOTH, expand=True, pady=6)

        left = ttk.Frame(area)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        ttk.Label(left, text="1) Entradas (definidas por el JSON)").pack(anchor="w")
        self.lbl_info = ttk.Label(left, text="Cargue un JSON para crear los campos de entrada.")
        self.lbl_info.pack(anchor="w", pady=(0,6))

        # Contenedor donde se crearán los Entry de entradas x1..xn
        self.inputs_container = ttk.LabelFrame(left, text="Campos de entrada (manuales)", padding=6)
        self.inputs_container.pack(fill=tk.X, pady=6)

        # controles para patrones manuales
        ttk.Label(left, text="Patrones manuales añadidos:").pack(anchor="w", pady=(8,2))
        self.txt_manuales = tk.Text(left, height=8)
        self.txt_manuales.pack(fill=tk.BOTH, expand=False)
        self.txt_manuales.configure(state="disabled")

        right = ttk.Frame(area)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        ttk.Label(right, text="2) Resumen del JSON cargado").pack(anchor="w")
        self.txt_resumen = tk.Text(right, height=12)
        self.txt_resumen.pack(fill=tk.BOTH, expand=False)
        self.txt_resumen.configure(state="disabled")

        ttk.Label(right, text="3) Resultados / Métricas").pack(anchor="w", pady=(8,0))
        self.txt_resultados = tk.Text(right, height=20)
        self.txt_resultados.pack(fill=tk.BOTH, expand=True)
        self.txt_resultados.configure(state="disabled")

    def _set_text(self, widget, txt):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, txt)
        widget.configure(state="disabled")

    # -------------------------------------------------------
    # Cargar JSON y preparar inputs según 'resumen' o centros
    # -------------------------------------------------------
    def _cargar_json(self):
        path = filedialog.askopenfilename(parent=self.parent, title="Seleccionar JSON del entrenamiento", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el JSON:\n{e}", parent=self.parent)
            return

        self.model_json = j
        # extraer datos
        self.centros = np.array(j.get("centros_radiales")) if j.get("centros_radiales") is not None else None
        self.distancias = np.array(j.get("distancias")) if j.get("distancias") is not None else None
        self.fa_train = np.array(j.get("funcion_activacion")) if j.get("funcion_activacion") is not None else None
        self.matriz_A = np.array(j.get("matriz_interpolacion")) if j.get("matriz_interpolacion") is not None else None
        self.pesos = _pesos_from_json(j.get("pesos")) if j.get("pesos") is not None else None
        self.resumen = j.get("resumen", {}) if isinstance(j.get("resumen", {}), dict) else {}

        # -------------- INPUT NAMES (PRIMARIO) -----------------
        input_names = None
        if isinstance(j.get("input_names"), (list, tuple)):
            input_names = [str(x) for x in j.get("input_names")]
        elif isinstance(self.resumen.get("columns"), (list, tuple)):
            cols = list(self.resumen.get("columns"))
            if len(cols) >= 1:
                input_names = [str(c) for c in cols[:-1]] if len(cols) > 1 else [str(cols[0])]

        # fallback: inferir desde resumen o centros
        if input_names is None:
            n_inputs = None
            if "entradas" in self.resumen:
                try:
                    n_inputs = int(self.resumen["entradas"])
                except Exception:
                    n_inputs = None
            if n_inputs is None and self.centros is not None:
                try:
                    if self.centros.ndim == 2:
                        n_inputs = int(self.centros.shape[1])
                except Exception:
                    n_inputs = None
            if n_inputs is None:
                n_inputs = 1
            input_names = [f"x{i+1}" for i in range(n_inputs)]

        # guardar y crear entradas visuales
        self.input_names = input_names
        self.n_inputs = len(self.input_names)
        self._crear_inputs_entries(self.input_names)

        # mostrar resumen compacto (añadimos input_names al texto)
        lines = ["JSON cargado correctamente.\n"]
        if isinstance(self.resumen, dict) and self.resumen:
            lines.append("Resumen:")
            for k in ("entradas", "salidas", "patrones", "error_optimo", "num_centros"):
                if k in self.resumen:
                    lines.append(f"  {k}: {self.resumen[k]}")
            lines.append("")
        lines.append(f"Nombres de entradas detectados: {self.input_names}")
        if self.centros is not None:
            lines.append(f"Centros radiales: shape={self.centros.shape}")
        if self.matriz_A is not None:
            lines.append(f"Matriz de interpolación (A): shape={self.matriz_A.shape}")
        if self.distancias is not None:
            lines.append(f"Distancias guardadas: shape={self.distancias.shape}")
        if self.pesos is not None:
            lines.append(f"Pesos: len={self.pesos.size}")
        if "sigma" in j:
            try:
                self.sigma = float(j["sigma"])
                lines.append(f"Sigma (from JSON): {self.sigma}")
            except Exception:
                pass

        self._set_text(self.txt_resumen, "\n".join(lines))
        messagebox.showinfo("JSON cargado", "Modelo cargado. Complete las entradas y agregue patrones manuales para simular.", parent=self.parent)

    def _crear_inputs_entries(self, names):
        for w in self.inputs_container.winfo_children():
            w.destroy()
        self.input_entries = []
        per_row = 4
        row = None
        for i, nm in enumerate(names):
            if i % per_row == 0:
                row = ttk.Frame(self.inputs_container)
                row.pack(fill=tk.X, pady=2)
            f = ttk.Frame(row)
            f.pack(side=tk.LEFT, padx=6)
            label_txt = nm
            ttk.Label(f, text=label_txt).pack(anchor="w")
            e = ttk.Entry(f, width=12)
            e.pack()
            self.input_entries.append(e)
        self.lbl_info.config(text=f"Campos creados: {len(self.input_entries)} entradas (puedes ingresar valores y añadir patrones).")

    def _agregar_patron_manual_from_entries(self):
        if not self.input_entries:
            messagebox.showwarning("Sin JSON", "Carga primero un JSON para definir las entradas.")
            return
        vals = []
        try:
            for e in self.input_entries:
                txt = e.get().strip()
                if txt == "":
                    raise ValueError("Campos vacíos")
                vals.append(float(txt))
        except Exception:
            messagebox.showerror("Valor inválido", "Asegúrate que todos los campos de entrada tengan un número válido.")
            return
        arr = np.array(vals, dtype=float)
        if arr.size != self.n_inputs:
            messagebox.showerror("Tamaño inválido", f"El patrón debe tener {self.n_inputs} valores.")
            return
        self.manual_patterns.append(arr)
        self.txt_manuales.configure(state="normal")
        self.txt_manuales.insert(tk.END, ", ".join([f"{x:.6g}" for x in arr]) + "\n")
        self.txt_manuales.configure(state="disabled")
        for e in self.input_entries:
            e.delete(0, tk.END)

    def _limpiar_manuales(self):
        self.manual_patterns = []
        self.txt_manuales.configure(state="normal")
        self.txt_manuales.delete("1.0", tk.END)
        self.txt_manuales.configure(state="disabled")

    # -------------------------------------------------------
    # Ejecutar simulación (sobre patrones manuales)
    # -------------------------------------------------------
    def _ejecutar_simulacion(self, manuales_only=False):
        if self.model_json is None:
            messagebox.showwarning("Sin modelo", "Carga primero el JSON del entrenamiento.")
            return

        if manuales_only and len(self.manual_patterns) == 0:
            messagebox.showwarning("Sin patrones", "No hay patrones manuales añadidos.")
            return

        if len(self.manual_patterns) == 0:
            messagebox.showinfo("Necesita patrones", "No hay patrones manuales. Ingresa valores y pulsa 'Agregar patrón desde entradas'.")
            return

        X_pred = np.vstack(self.manual_patterns)  # (n_pred, n_inputs)
        n_pred = X_pred.shape[0]

        # verificar centros y pesos
        if self.centros is None:
            messagebox.showerror("Centros faltantes", "El JSON no contiene 'centros_radiales'. No puedo calcular la FA para nuevos patrones.")
            return
        if self.pesos is None:
            messagebox.showerror("Pesos faltantes", "El JSON no contiene 'pesos'. No puedo calcular la salida.")
            return

        centros = np.asarray(self.centros, dtype=float)  # (n_centros, n_inputs)
        n_centros = centros.shape[0]

        # calcular distancias D (n_pred, n_centros)
        try:
            dif = X_pred[:, np.newaxis, :] - centros[np.newaxis, :, :]  # (n_pred, n_centros, n_inputs)
            D = np.sqrt(np.sum(dif**2, axis=2))  # (n_pred, n_centros)
        except Exception as e:
            messagebox.showerror("Error distancias", f"No se pudo calcular las distancias:\n{e}")
            return

        # calcular FA = D^2 * ln(D), con manejo de D==0
        eps = 1e-12
        FA = np.zeros_like(D, dtype=float)
        mask = D > 0
        # usar ln(D) directo; D>0 garantizado por mask
        FA[mask] = (D[mask]**2) * np.log(D[mask])
        # si algún D muy pequeño produce -inf o nan, reemplazar por 0
        FA = np.nan_to_num(FA, nan=0.0, posinf=0.0, neginf=0.0)

        # pesos
        W = np.atleast_1d(self.pesos).reshape(-1)
        # NOTA: se asume orden W0, W1, W2, ... (W0 -> bias opcional)
        # si W tiene len = n_centros + 1 -> asumimos W0 = bias, W1..Wm weights
        results_lines = []
        for i in range(n_pred):
            d_vec = D[i, :]    # distancias a cada centro
            fa_vec = FA[i, :]  # FA por centro

            # calcular YR segun cantidad de pesos
            YR = None
            equation_parts = []

            if W.size == n_centros + 1:
                bias = float(W[0])
                weights = W[1:]
                # YR = bias + sum(weights * fa_vec)
                YR_val = bias + float(np.dot(weights, fa_vec))
                YR = float(YR_val)
                # construir representación textual de la ecuacion:
                equation_parts.append(f"{bias:.6g}*1")
                for j, (wj, fj, dj) in enumerate(zip(weights, fa_vec, d_vec), start=1):
                    # mostrar FA numérico (puede ser negativo)
                    equation_parts.append(f"{wj:.6g}*({fj:.6g})")
            elif W.size == n_centros:
                weights = W
                YR_val = float(np.dot(weights, fa_vec))
                YR = YR_val
                for j, (wj, fj, dj) in enumerate(zip(weights, fa_vec, d_vec), start=1):
                    equation_parts.append(f"{wj:.6g}*({fj:.6g})")
            else:
                # intentar adaptarse si W está ordenado con bias al final u otras variantes
                # caso bias al final
                if W.size == n_centros + 1:
                    bias = float(W[-1])
                    weights = W[:-1]
                    YR = float(np.dot(weights, fa_vec) + bias)
                    equation_parts.append(f"{bias:.6g}*1")
                    for j, (wj, fj, dj) in enumerate(zip(weights, fa_vec), start=1):
                        equation_parts.append(f"{wj:.6g}*({fj:.6g})")
                else:
                    # no compatible
                    messagebox.showerror("Pesos incompatibles",
                        f"El número de pesos en JSON ({W.size}) es incompatible con n_centros ({n_centros}).")
                    return

            # componer texto para este patrón
            lines = []
            lines.append(f"Patrón {i+1}:")
            # mostrar vector X para claridad
            lines.append("  Entradas X = [" + ", ".join([f"{v:.6g}" for v in X_pred[i]]) + "]")
            lines.append("  Distancias D_j  = [" + ", ".join([f"{v:.6g}" for v in d_vec]) + "]")
            lines.append("  FA_j = D_j^2 * ln(D_j) = [" + ", ".join([f"{v:.6g}" for v in fa_vec]) + "]")
            # ecuación textual
            eq_text = " + ".join(equation_parts)
            lines.append("  Ecuación (sustitución): " + eq_text)
            lines.append(f"  Resultado YR = {YR:.6g}")
            lines.append("")  # separación
            results_lines.extend(lines)

        # mostrar resultado en la ventana
        self._set_text(self.txt_resultados, "\n".join(results_lines))

        # opcional: graficar YR si quieres (aquí solo mostramos YR en lista)
        try:
            YR_list = []
            for i in range(n_pred):
                # recomputar YR para graficar correctamente (repetimos la misma lógica, pero ya calculado arriba)
                # preferimos extraer del text, pero recomputamos:
                fa_vec = FA[i, :]
                if W.size == n_centros + 1:
                    YR_list.append(float(W[0] + np.dot(W[1:], fa_vec)))
                elif W.size == n_centros:
                    YR_list.append(float(np.dot(W, fa_vec)))
                else:
                    # fallback
                    if W.size == n_centros + 1:
                        YR_list.append(float(W[-1] + np.dot(W[:-1], fa_vec)))
                    else:
                        YR_list.append(float(np.dot(W[:min(W.size, n_centros)], fa_vec)))
            # graficar YR (solo para inspección)
            if len(YR_list) > 0:
                plt.figure(figsize=(6,4))
                plt.plot(np.arange(1, len(YR_list)+1), YR_list, marker='o', linestyle='-')
                plt.title("YR (predicciones) sobre patrones manuales")
                plt.xlabel("Patrón")
                plt.ylabel("YR")
                plt.grid(True)
                plt.show()
        except Exception:
            # no fatal
            pass

# función de conveniencia para la app principal:
def launch_simulation_panel(parent_container, app_ref=None):
    SimulacionPanel(parent_container, app_ref)
