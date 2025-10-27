# simulacion_rbf.py
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _pesos_from_json(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return np.array([v for _, v in items], dtype=float)
    return np.array(obj, dtype=float)

def _inferir_sigma_robusto(centros, distancias=None, fa=None):
    # Intenta estimar sigma desde distancias/FA; si falla, heurística NN; fallback 1.0
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
    # nearest-neighbour heuristic on centers
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
    Panel de simulación embebido en la ventana principal.
    parent_container: frame donde se montará el panel (por ejemplo self.content)
    app_ref: referencia a la instancia RBFApp (para acceder a current_train/current_test/summary)
    """
    def __init__(self, parent_container, app_ref):
        self.parent = parent_container
        self.app = app_ref
        # limpiar container y montar panel
        for w in self.parent.winfo_children():
            w.destroy()

        self.frame = ttk.Frame(self.parent, padding=8)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Estado interno
        self.dataset = None
        self.df_test = None
        self.df_train = None
        self.n_inputs = 0
        self.manual_patterns = []  # lista de arrays
        self.model_json = None
        self.centros = None
        self.distancias = None
        self.fa_train = None
        self.matriz_A = None
        self.pesos = None
        self.resumen = None
        self.sigma = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.frame)
        top.pack(fill=tk.X, pady=4)

        ttk.Button(top, text="Cargar dataset (CSV/XLSX)", command=self._cargar_dataset).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Cargar JSON (modelo entrenado)", command=self._cargar_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Simular (usar manuales si hay)", command=self._ejecutar_simulacion).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Limpiar patrones manuales", command=self._limpiar_manuales).pack(side=tk.LEFT, padx=4)

        # area central: left = entradas manuales / preview ; right = resumen modelo y resultados
        area = ttk.Frame(self.frame)
        area.pack(fill=tk.BOTH, expand=True, pady=6)

        left = ttk.Frame(area)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        ttk.Label(left, text="1) Dataset / Entradas").pack(anchor="w")
        self.lbl_dataset_info = ttk.Label(left, text="No hay dataset cargado.")
        self.lbl_dataset_info.pack(anchor="w", pady=(0,6))

        self.manual_frame = ttk.LabelFrame(left, text="Ingresar patrón manual (entradas)", padding=6)
        self.manual_frame.pack(fill=tk.X, pady=6)

        # aquí se generarán los labels+entry para cada entrada (cuando carguemos dataset o JSON)
        self.inputs_container = ttk.Frame(self.manual_frame)
        self.inputs_container.pack(fill=tk.X)

        btns = ttk.Frame(self.manual_frame)
        btns.pack(fill=tk.X, pady=(6,0))
        ttk.Button(btns, text="Agregar patrón manual", command=self._agregar_patron_manual).pack(side=tk.LEFT)
        ttk.Button(btns, text="Simular patrones manuales", command=lambda: self._ejecutar_simulacion(manuales_only=True)).pack(side=tk.LEFT, padx=6)

        ttk.Label(left, text="Patrones manuales añadidos:").pack(anchor="w", pady=(8,2))
        self.txt_manuales = tk.Text(left, height=6)
        self.txt_manuales.pack(fill=tk.BOTH, expand=False)
        self.txt_manuales.configure(state="disabled")

        # derecho: resumen JSON y resultados
        right = ttk.Frame(area)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        ttk.Label(right, text="2) Resumen del modelo (JSON)").pack(anchor="w")
        self.txt_resumen = tk.Text(right, height=12)
        self.txt_resumen.pack(fill=tk.BOTH, expand=False)
        self.txt_resumen.configure(state="disabled")

        ttk.Label(right, text="3) Resultados de simulación").pack(anchor="w", pady=(8,0))
        self.txt_resultados = tk.Text(right, height=16)
        self.txt_resultados.pack(fill=tk.BOTH, expand=True)
        self.txt_resultados.configure(state="disabled")

    # ---------- dataset ----------
    def _cargar_dataset(self):
        path = filedialog.askopenfilename(parent=self.parent, title="Seleccionar dataset", filetypes=[("CSV","*.csv"),("Excel","*.xlsx;*.xls")])
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el dataset:\n{e}", parent=self.parent)
            return

        # guardar dataset en panel (no modificamos el training app aquí)
        self.dataset = df.reset_index(drop=True)
        n_cols = df.shape[1]
        if n_cols < 2:
            messagebox.showerror("Error", "El dataset debe tener al menos 2 columnas (entradas + salida).", parent=self.parent)
            return
        self.n_inputs = n_cols - 1

        # obtener nombres reales de las columnas de entrada (toma todas menos la última)
        # si los nombres no son string los convertimos
        colnames = [str(c) for c in df.columns[:self.n_inputs]]
        self.input_names = colnames

        # crear widgets para ingresar entradas manuales usando los nombres reales
        self._crear_inputs_entries(self.input_names)

        self.lbl_dataset_info.config(text=f"Dataset cargado: '{Path(path).name}' — entradas = {self.n_inputs}, filas = {len(df)}")
        messagebox.showinfo("Dataset cargado", f"Dataset cargado: {len(df)} filas. Ahora puedes ingresar patrones manuales.")

    def _crear_inputs_entries(self, names):
        """
        names: lista de strings con los nombres de las columnas de entrada (longitud = n_inputs)
        Crea un Entry por cada nombre y pone el label con el nombre real.
        """
        # destruir hijos previos
        for w in self.inputs_container.winfo_children():
            w.destroy()
        self.input_entries = []

        # organizamos en filas con hasta 4 entradas por fila para que no se desborde
        per_row = 4
        for i, nm in enumerate(names):
            # nueva fila cada `per_row` elementos
            if i % per_row == 0:
                row = ttk.Frame(self.inputs_container)
                row.pack(fill=tk.X, pady=2)
            f = ttk.Frame(row)
            f.pack(side=tk.LEFT, padx=6)
            # etiqueta con el nombre original (acorta si es muy largo)
            label_txt = nm if len(nm) <= 18 else (nm[:15] + "...")
            ttk.Label(f, text=label_txt).pack(anchor="w")
            e = ttk.Entry(f, width=14)
            e.pack()
            # tooltip / full name en bind (si el nombre fue truncado, al pasar el mouse se muestra completo)
            if len(nm) > 18:
                def _on_enter(event, full=nm):
                    event.widget.master.children[list(event.widget.master.children.keys())[0]].configure(text=full)
                def _on_leave(event, short=label_txt):
                    event.widget.master.children[list(event.widget.master.children.keys())[0]].configure(text=short)
                # no todos los widgets soportan tooltips fáciles; como alternativa dejamos el label completo en title al hacer click
                e.bind("<FocusIn>", lambda ev, full=nm: e.delete(0, tk.END) )
            self.input_entries.append(e)


    def _agregar_patron_manual(self):
        if self.n_inputs == 0:
            messagebox.showwarning("Sin entradas", "Primero cargue un dataset para conocer el número de entradas.")
            return
        vals = []
        try:
            for e in self.input_entries:
                v = float(e.get().strip())
                vals.append(v)
        except Exception:
            messagebox.showerror("Valor inválido", "Asegúrate que todas las entradas manuales sean números válidos.")
            return
        arr = np.array(vals, dtype=float)
        self.manual_patterns.append(arr)
        # actualizar listado
        self.txt_manuales.configure(state="normal")
        self.txt_manuales.insert(tk.END, ", ".join([f"{x:.6g}" for x in arr]) + "\n")
        self.txt_manuales.configure(state="disabled")
        # limpiar campos
        for e in self.input_entries:
            e.delete(0, tk.END)

    def _limpiar_manuales(self):
        self.manual_patterns = []
        self.txt_manuales.configure(state="normal")
        self.txt_manuales.delete("1.0", tk.END)
        self.txt_manuales.configure(state="disabled")

    # ---------- JSON ----------
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
        self.centros = np.array(j.get("centros_radiales")) if j.get("centros_radiales") is not None else None
        self.distancias = np.array(j.get("distancias")) if j.get("distancias") is not None else None
        self.fa_train = np.array(j.get("funcion_activacion")) if j.get("funcion_activacion") is not None else None
        self.matriz_A = np.array(j.get("matriz_interpolacion")) if j.get("matriz_interpolacion") is not None else None
        self.pesos = _pesos_from_json(j.get("pesos")) if j.get("pesos") is not None else None
        self.resumen = j.get("resumen", {})

        # mostrar resumen compacto
        lines = ["Modelo JSON cargado.\n"]
        if isinstance(self.resumen, dict):
            lines.append("Resumen guardado:")
            for k in ("entradas", "salidas", "patrones", "error_optimo", "num_centros"):
                if k in self.resumen:
                    lines.append(f"  {k}: {self.resumen[k]}")
            lines.append("")
        if self.centros is not None:
            lines.append(f"Centros radiales (shape {self.centros.shape})")
        if self.matriz_A is not None:
            lines.append(f"Matriz A (shape {self.matriz_A.shape})")
        if self.pesos is not None:
            lines.append(f"Pesos (len {self.pesos.size})")
        self._set_text(self.txt_resumen, "\n".join(lines))

        messagebox.showinfo("JSON cargado", "Modelo cargado desde JSON. Puede ahora simular con este modelo.")

    # ---------- Simulación ----------
    def _ejecutar_simulacion(self, manuales_only=False):
        """
        Si manuales_only=True simula solo sobre manual_patterns;
        si manuales_only=False: si hay manuales usa ellos, sino intenta simular sobre el dataset cargado (toma todas las filas).
        """
        # validar que haya modelo (pesos/centros)
        if self.pesos is None and self.matriz_A is None:
            messagebox.showwarning("Sin modelo", "Cargue el JSON del entrenamiento (pesos o matriz A).")
            return

        # decidir datos X sobre los cuales predecir
        if manuales_only and len(self.manual_patterns) == 0:
            messagebox.showwarning("Sin patrones manuales", "No hay patrones manuales agregados.")
            return

        if len(self.manual_patterns) > 0 and (not manuales_only):
            # preferimos usar manuales si existen
            X_pred = np.vstack(self.manual_patterns)
            Yd_pred = None  # no hay Yd para manuales
            source = "manuales"
        elif manuales_only:
            X_pred = np.vstack(self.manual_patterns)
            Yd_pred = None
            source = "manuales"
        else:
            # usar dataset cargado (si existe)
            if self.dataset is None:
                messagebox.showwarning("Sin datos", "No hay dataset ni patrones manuales para simular.")
                return
            n_inputs = self.n_inputs
            X_pred = self.dataset.iloc[:, :n_inputs].to_numpy(dtype=float)
            Yd_pred = self.dataset.iloc[:, n_inputs].to_numpy(dtype=float)
            source = "dataset"

        # Si tenemos matriz_A guardada y su número de filas coincide con X_pred -> usarla directamente
        A_pred = None
        if self.matriz_A is not None:
            try:
                Aarr = np.asarray(self.matriz_A, dtype=float)
                if Aarr.shape[0] == X_pred.shape[0]:
                    A_pred = Aarr
                else:
                    A_pred = None
            except Exception:
                A_pred = None

        # si A_pred no está, necesitamos centros y sigma para calcular FA
        if A_pred is None:
            if self.centros is None:
                messagebox.showwarning("Faltan centros", "No se encontraron centros en el JSON para calcular FA. Asegúrate que el JSON contenga 'centros_radiales'.")
                return
            sigma = _inferir_sigma_robusto(self.centros, self.distancias, self.fa_train)
            self.sigma = sigma
            # calcular FA_test (A_pred)
            try:
                dif = X_pred[:, np.newaxis, :] - self.centros[np.newaxis, :, :]
                D2 = np.sum(dif**2, axis=2)
                denom = 2.0 * (sigma**2)
                A_pred = np.exp(-D2 / denom)
            except Exception as e:
                messagebox.showerror("Error FA", f"No se pudo calcular la función de activación para los datos:\n{e}")
                return

        # ahora W
        W = _pesos_from_json(self.pesos) if self.pesos is not None else None
        if W is None:
            messagebox.showwarning("Faltan pesos", "No se encontraron pesos en el JSON.")
            return
        W = np.atleast_1d(W).reshape(-1)

        # calcular YR con manejo bias/multisalida
        try:
            if W.ndim == 2 and W.shape[1] == 1:
                W = W.ravel()

            if W.size == A_pred.shape[1]:
                YR = A_pred.dot(W)
            elif W.size == A_pred.shape[1] + 1:
                bias = W[-1]
                w_use = W[:-1]
                YR = A_pred.dot(w_use) + bias
            elif W.size % A_pred.shape[1] == 0:
                K = W.size // A_pred.shape[1]
                W_mat = W.reshape(A_pred.shape[1], K)
                out = A_pred.dot(W_mat)
                YR = out[:, 0] if out.ndim == 2 and out.shape[1] >= 1 else out.ravel()
            else:
                raise ValueError(f"Dimensiones incompatibles: A cols={A_pred.shape[1]} vs len(W)={W.size}.")
        except Exception as e:
            messagebox.showerror("Error predicción", f"No se pudo calcular YR por incompatibilidad de dimensiones:\n{e}")
            return

        # calcular EL/EG/MAE/RMSE si Yd_pred existe
        text_lines = []
        text_lines.append(f"Simulación sobre: {source} (n={len(YR)})")
        if Yd_pred is not None:
            n = min(len(YR), len(Yd_pred))
            YR = np.asarray(YR).ravel()[:n]
            Yd_pred = np.asarray(Yd_pred).ravel()[:n]
            EL = Yd_pred - YR
            absEL = np.abs(EL)
            EG = float(np.mean(absEL))
            MAE = EG
            RMSE = float(np.sqrt(np.mean((Yd_pred - YR)**2)))
            text_lines.append(f"EG = {EG:.6g} (MAE)")
            text_lines.append(f"MAE = {MAE:.6g}")
            text_lines.append(f"RMSE = {RMSE:.6g}")
        else:
            text_lines.append("No hay Yd (patrones manuales); solo se muestran YR calculadas.")

        # Mostrar primeras predicciones
        text_lines.append("\nPrimeros resultados (YR):")
        for i in range(min(50, len(YR))):
            line = f"{i+1}: YR = {YR[i]:.6g}"
            if Yd_pred is not None:
                line += f" | YD = {Yd_pred[i]:.6g} | EL = { (Yd_pred[i]-YR[i]):.6g }"
            text_lines.append(line)

        self._set_text(self.txt_resultados, "\n".join(text_lines))

        # Mostrar gráficas (YD vs YR y dispersión) si hay Yd
        if Yd_pred is not None:
            try:
                fig, axs = plt.subplots(2,1, figsize=(8,10))
                # Yd vs Yr por patrón
                axs[0].plot(np.arange(1, len(YR)+1), Yd_pred, label="YD", marker='o')
                axs[0].plot(np.arange(1, len(YR)+1), YR, label="YR", marker='x')
                axs[0].set_xlabel("Patrón")
                axs[0].set_ylabel("Salida")
                axs[0].legend()
                axs[0].grid(True)

                # dispersión
                axs[1].scatter(Yd_pred, YR, s=20)
                mn = min(np.min(Yd_pred), np.min(YR))
                mx = max(np.max(Yd_pred), np.max(YR))
                axs[1].plot([mn,mx],[mn,mx], linestyle="--")
                axs[1].set_xlabel("YD")
                axs[1].set_ylabel("YR")
                axs[1].set_title("Dispersión YD vs YR")
                axs[1].grid(True)

                plt.tight_layout()
                plt.show()
            except Exception as e:
                messagebox.showwarning("Gráficas", f"No se pudieron generar las gráficas:\n{e}")

    # helpers
    def _set_text(self, widget, txt):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, txt)
        widget.configure(state="disabled")


# función de conveniencia para la app principal:
def launch_simulation_panel(parent_container, app_ref):
    """
    parent_container: frame donde montar la vista de simulación (por ejemplo self.content)
    app_ref: referencia a la instancia RBFApp (si necesitas compartir estado)
    """
    SimulacionPanel(parent_container, app_ref)
