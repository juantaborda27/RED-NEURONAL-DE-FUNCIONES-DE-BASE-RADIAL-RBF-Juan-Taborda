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
        # ordenar por clave (W0, W1, ...) para mantener el orden
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return np.array([v for _, v in items], dtype=float)
    return np.array(obj, dtype=float)

def _inferir_sigma_robusto(centros, distancias=None, fa=None):
    """
    Intenta estimar sigma (desviación) de varias formas:
    1) usando distancias y FA si FA ~ exp(-D^2/(2*sigma^2)) (si hay info)
    2) heurística nearest-neighbour entre centros (mediana de distancias NN / sqrt(2 ln 2))
    3) fallback = 1.0
    """
    try:
        if distancias is not None and fa is not None:
            D = np.asarray(distancias, dtype=float)
            F = np.asarray(fa, dtype=float)
            # intentar obtener relaciones donde 0 < F < 1 y D > 0
            mask = (F > 0) & (F < 1) & (D > 0)
            if np.any(mask):
                # supongamos F = exp(-D^2/(2 sigma^2)) => sigma^2 = -D^2/(2 ln F)
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
            # distancia euclidiana entre centros
            D2 = np.sum((C[:, None, :] - C[None, :, :])**2, axis=2)
            np.fill_diagonal(D2, np.inf)
            D = np.sqrt(D2)
            nn = np.min(D, axis=1)
            nn = nn[np.isfinite(nn)]
            if nn.size > 0:
                d_med = float(np.median(nn))
                if d_med > 0:
                    # heurística: sigma ~ median_nn / sqrt(2 ln 2)
                    return float(d_med / np.sqrt(2.0 * np.log(2.0)))
    except Exception:
        pass

    return 1.0

class SimulacionPanel:
    """
    Panel de simulación sin uso de dataset: todo parte del JSON cargado.
    parent_container: frame donde se montará el panel (por ejemplo self.content)
    app_ref: referencia a la instancia RBFApp (solo si quieres acceso a estado global, no obligatorio)
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
        # 1) si el JSON tiene explicitamente input_names (lo guardamos así)
        if isinstance(j.get("input_names"), (list, tuple)):
            input_names = [str(x) for x in j.get("input_names")]
        # 2) si no, si tenemos columns en resumen -> usamos todas menos la última
        elif isinstance(self.resumen.get("columns"), (list, tuple)):
            cols = list(self.resumen.get("columns"))
            if len(cols) >= 1:
                input_names = [str(c) for c in cols[:-1]] if len(cols) > 1 else [str(cols[0])]
        # 3) fallback: inferir desde centros (si existen)
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
        # limpiar contenedor anterior
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

        # actualizar info
        self.lbl_info.config(text=f"Campos creados: {len(self.input_entries)} entradas (puedes ingresar valores y añadir patrones).")

    # -------------------------------------------------------
    # Agregar patrón manual desde los entries actuales
    # -------------------------------------------------------
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
        # actualizar listado
        self.txt_manuales.configure(state="normal")
        self.txt_manuales.insert(tk.END, ", ".join([f"{x:.6g}" for x in arr]) + "\n")
        self.txt_manuales.configure(state="disabled")
        # limpiar entries
        for e in self.input_entries:
            e.delete(0, tk.END)

    def _limpiar_manuales(self):
        self.manual_patterns = []
        self.txt_manuales.configure(state="normal")
        self.txt_manuales.delete("1.0", tk.END)
        self.txt_manuales.configure(state="disabled")

    # -------------------------------------------------------
    # Ejecutar simulación (solo manuales o todos los manuales)
    # -------------------------------------------------------
    def _ejecutar_simulacion(self, manuales_only=False):
        if self.model_json is None:
            messagebox.showwarning("Sin modelo", "Carga primero el JSON del entrenamiento.")
            return

        if manuales_only and len(self.manual_patterns) == 0:
            messagebox.showwarning("Sin patrones", "No hay patrones manuales añadidos.")
            return

        if len(self.manual_patterns) > 0:
            X_pred = np.vstack(self.manual_patterns)
            Yd_pred = None
            source = "manuales"
        else:
            # sin dataset disponible, obligamos a tener al menos un patrón en entradas
            messagebox.showinfo("Necesita patrones", "No hay patrones manuales. Ingresa valores en las entradas y pulsa 'Agregar patrón desde entradas'.")
            return

        # obtener A_pred: si matriz_A coincide (no aplicable aquí) calculamos usando centros y sigma
        if self.matriz_A is not None and False:
            # mantenemos la condición but normalmente matriz_A es para train patterns y no se usa aquí
            A_pred = np.asarray(self.matriz_A, dtype=float)
        else:
            if self.centros is None:
                messagebox.showerror("Centros faltantes", "El JSON no contiene 'centros_radiales'. No puedo calcular la FA para nuevos patrones.")
                return
            # inferir sigma si no existe
            if self.sigma is None:
                self.sigma = _inferir_sigma_robusto(self.centros, self.distancias, self.fa_train)
            # calcular D^2
            try:
                dif = X_pred[:, np.newaxis, :] - self.centros[np.newaxis, :, :]
                D2 = np.sum(dif**2, axis=2)  # (n_pred, n_centros)
                denom = 2.0 * (self.sigma ** 2)
                # función RBF (gaussiana)
                A_pred = np.exp(-D2 / denom)
            except Exception as e:
                messagebox.showerror("Error FA", f"No se pudo calcular la función de activación (A):\n{e}")
                return

        # pesos
        W = _pesos_from_json(self.model_json.get("pesos")) if self.model_json.get("pesos") is not None else None
        if W is None:
            messagebox.showerror("Pesos faltantes", "No se encontraron pesos en el JSON.")
            return
        W = np.atleast_1d(W).reshape(-1)

        # calcular YR: manejar bias si está como peso extra
        try:
            if W.size == A_pred.shape[1]:
                YR = A_pred.dot(W)
            elif W.size == A_pred.shape[1] + 1:
                # asumir último elemento bias
                bias = W[-1]
                w_use = W[:-1]
                YR = A_pred.dot(w_use) + bias
            elif W.size % A_pred.shape[1] == 0:
                # multi-salida (W arranged as n_centros * n_outputs)
                K = W.size // A_pred.shape[1]
                W_mat = W.reshape(A_pred.shape[1], K)
                out = A_pred.dot(W_mat)
                # si hubo múltiples salidas, mostramos la primera columna por simplicidad
                YR = out[:, 0] if out.ndim == 2 and out.shape[1] >= 1 else out.ravel()
            else:
                raise ValueError(f"Dimensiones incompatibles: A cols={A_pred.shape[1]} vs len(W)={W.size}")
        except Exception as e:
            messagebox.showerror("Error predicción", f"No se pudo calcular YR:\n{e}")
            return

        # métricas (no hay Yd para manuales)
        lines = []
        lines.append(f"Simulación sobre: {source} (n={len(YR)})")
        if Yd_pred is not None:
            n = min(len(YR), len(Yd_pred))
            YR = np.asarray(YR).ravel()[:n]
            Yd_pred = np.asarray(Yd_pred).ravel()[:n]
            EL = Yd_pred - YR
            absEL = np.abs(EL)
            EG = float(np.mean(absEL))
            MAE = EG
            RMSE = float(np.sqrt(np.mean((Yd_pred - YR)**2)))
            lines.append(f"EG = {EG:.6g} (MAE)")
            lines.append(f"MAE = {MAE:.6g}")
            lines.append(f"RMSE = {RMSE:.6g}")
        else:
            lines.append("No hay Yd (patrones manuales). Se muestran YR calculadas.")

        lines.append("\nPrimeros resultados (YR):")
        for i in range(min(50, len(YR))):
            line = f"{i+1}: YR = {YR[i]:.6g}"
            lines.append(line)

        # agregar sigma y shape para debug
        lines.append("\n--- Info interna ---")
        lines.append(f"sigma used = {self.sigma:.6g}")
        lines.append(f"n_centros = {self.centros.shape[0] if self.centros is not None else 'N/A'}, "
                     f"A_pred shape = {A_pred.shape}, W len = {W.size}")

        self._set_text(self.txt_resultados, "\n".join(lines))

        # gráficas si hubiese Yd_pred (no aplicable para manuales)
        if Yd_pred is not None:
            try:
                fig, axs = plt.subplots(2, 1, figsize=(8, 10))
                axs[0].plot(np.arange(1, len(YR)+1), Yd_pred, label="YD", marker='o')
                axs[0].plot(np.arange(1, len(YR)+1), YR, label="YR", marker='x')
                axs[0].set_xlabel("Patrón")
                axs[0].set_ylabel("Salida")
                axs[0].legend()
                axs[0].grid(True)

                axs[1].scatter(Yd_pred, YR, s=20)
                mn = min(np.min(Yd_pred), np.min(YR))
                mx = max(np.max(Yd_pred), np.max(YR))
                axs[1].plot([mn, mx], [mn, mx], linestyle="--")
                axs[1].set_xlabel("YD")
                axs[1].set_ylabel("YR")
                axs[1].set_title("Dispersión YD vs YR")
                axs[1].grid(True)

                plt.tight_layout()
                plt.show()
            except Exception as e:
                messagebox.showwarning("Gráficas", f"No se pudieron generar las gráficas:\n{e}")

# función de conveniencia para la app principal:
def launch_simulation_panel(parent_container, app_ref=None):
    SimulacionPanel(parent_container, app_ref)
