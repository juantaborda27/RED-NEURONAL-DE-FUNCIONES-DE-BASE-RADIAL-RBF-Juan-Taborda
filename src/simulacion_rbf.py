# simulacion_rbf.py
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from pathlib import Path

class SimuladorRBF:
    """
    Clase que contiene la lógica de simulación y una GUI ligera en una ventana Toplevel.
    Diseñada para ser llamada desde la app principal.
    """
    def __init__(self, master):
        self.master = master
        self.win = tk.Toplevel(master)
        self.win.title("Simulación RBF")
        self.win.geometry("900x900")
        self.win.transient(master)

        # Modelo cargado (desde JSON)
        self.modelo = None
        self.centros = None
        self.distancias = None
        self.fa_train = None
        self.matriz_A = None
        self.pesos = None
        self.resumen = None

        # Dataset de prueba
        self.dataset = None
        self.X = None
        self.Yd = None

        # sigma (spread) - se maneja internamente; NUNCA se pedirá al usuario
        self.sigma = None

        # último resultado
        self.ultimo_resultado = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        frm = ttk.Frame(self.win, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text="1) Modelo (JSON)").pack(anchor="w", pady=(2,4))
        ttk.Button(left, text="Cargar modelo (JSON)", command=self.cargar_modelo).pack(fill="x", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        ttk.Label(left, text="2) Dataset de prueba").pack(anchor="w", pady=(2,4))
        ttk.Button(left, text="Cargar dataset (CSV/XLSX)", command=self.cargar_dataset).pack(fill="x", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)

        # Nota: ya no hay botón para pedir sigma manual; sigma se infiere internamente o queda en 1.0.
        ttk.Button(left, text="Ejecutar simulación", command=self.ejecutar_simulacion).pack(fill="x", pady=8)

        # Panel derecho con resultados
        right = ttk.Frame(frm)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Label(right, text="Resultados:").pack(anchor="w")
        self.txt = tk.Text(right, height=30, wrap="none")
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.configure(state="disabled")

    # ---------- Helpers ----------
    def _pesos_from_json(self, obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: kv[0])
            return np.array([v for _, v in items], dtype=float)
        return np.array(obj, dtype=float)

    # ---------- Carga modelo ----------
    def cargar_modelo(self):
        path = filedialog.askopenfilename(parent=self.win, title="Seleccionar JSON del entrenamiento", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el JSON:\n{e}", parent=self.win)
            return

        self.modelo = j
        self.centros = np.array(j.get("centros_radiales")) if j.get("centros_radiales") is not None else None
        self.distancias = np.array(j.get("distancias")) if j.get("distancias") is not None else None
        self.fa_train = np.array(j.get("funcion_activacion")) if j.get("funcion_activacion") is not None else None
        self.matriz_A = np.array(j.get("matriz_interpolacion")) if j.get("matriz_interpolacion") is not None else None
        self.pesos = self._pesos_from_json(j.get("pesos")) if j.get("pesos") is not None else None
        self.resumen = j.get("resumen", {})

        # Mostrar información en el panel de texto (matriz A, pesos, YR calculado)
        display_lines = []
        display_lines.append("Modelo cargado desde JSON.\n")

        # Mostrar resumen corto si existe
        if isinstance(self.resumen, dict):
            display_lines.append("Resumen (extracto):")
            for k in ("entradas", "salidas", "patrones", "error_optimo", "num_centros"):
                if k in self.resumen:
                    display_lines.append(f"  {k}: {self.resumen[k]}")
            display_lines.append("")

        # Mostrar matriz de interpolación A si existe
        if self.matriz_A is not None:
            try:
                A = np.asarray(self.matriz_A, dtype=float)
                display_lines.append(f"Matriz de interpolación A (shape {A.shape}):")
                # mostrar primeras filas y columnas para no saturar
                maxr, maxc = min(10, A.shape[0]), min(10, A.shape[1])
                for i in range(maxr):
                    row_txt = "  " + ", ".join([f"{A[i,j]:.6g}" for j in range(maxc)])
                    if A.shape[1] > maxc:
                        row_txt += ", ..."
                    display_lines.append(row_txt)
                if A.shape[0] > maxr:
                    display_lines.append("  ...")
                display_lines.append("")
            except Exception as e:
                display_lines.append(f"[Error mostrando A: {e}]\n")
        else:
            display_lines.append("No se encontró la matriz de interpolación (A) en el JSON.\n")

        # Mostrar pesos si existen
        if self.pesos is not None:
            W = np.atleast_1d(np.asarray(self.pesos, dtype=float))
            display_lines.append(f"Pesos W (len {W.size}):")
            # mostrar algunos pesos
            w_show = ", ".join([f"W{i}={W[i]:.6g}" for i in range(min(10, W.size))])
            if W.size > 10:
                w_show += ", ..."
            display_lines.append("  " + w_show)
            display_lines.append("")
        else:
            display_lines.append("No se encontraron pesos en el JSON.\n")

        # Si A y W existen, calcular YR = A · W y mostrar ecuaciones reemplazadas (primeros N)
        if (self.matriz_A is not None) and (self.pesos is not None):
            try:
                A = np.asarray(self.matriz_A, dtype=float)
                W = np.atleast_1d(np.asarray(self.pesos, dtype=float))
                # ajustar forma si W fue guardado como (M,) o dict
                W = W.reshape(-1)
                # Intentamos alinear (si no coinciden, lo manejamos en ejecutar_simulacion)
                # Aquí simplemente mostramos A·W cuando posible (si alínean)
                if A.shape[1] == W.size:
                    YR_from_A = A.dot(W)
                    display_lines.append("YR calculado desde A · W (primeros valores):")
                    for i in range(min(10, YR_from_A.size)):
                        display_lines.append(f"  YR({i+1}) = {YR_from_A[i]:.6g}")
                    display_lines.append("")
                    # Mostrar ecuaciones con términos numéricos (primeras filas)
                    display_lines.append("Ecuaciones (primeras filas) con términos reemplazados:")
                    rows_disp = min(6, A.shape[0])
                    for i in range(rows_disp):
                        terms = []
                        for j in range(A.shape[1]):
                            aij = A[i, j]
                            wj = W[j]
                            terms.append(f"({aij:.6g}*{wj:.6g})")
                        sum_val = YR_from_A[i]
                        line = f"  YR({i+1}) = " + " + ".join(terms) + f" = {sum_val:.6g}"
                        display_lines.append(line)
                    display_lines.append("")
                else:
                    display_lines.append(f"[Aviso] A columns = {A.shape[1]} , W len = {W.size} -> no se calculó A·W aquí (lo intento al ejecutar simulación).\n")
            except Exception as e:
                display_lines.append(f"[Error al calcular A·W: {e}]\n")

        # Escribir todo en el text widget
        self.txt.configure(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "\n".join(display_lines))
        self.txt.configure(state="disabled")

        messagebox.showinfo("Modelo cargado", "Modelo RBF cargado correctamente.", parent=self.win)

    # ---------- Carga dataset ----------
    def cargar_dataset(self):
        path = filedialog.askopenfilename(parent=self.win, title="Seleccionar dataset", filetypes=[("CSV","*.csv"),("Excel","*.xlsx;*.xls")])
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el dataset:\n{e}", parent=self.win)
            return

        if df.shape[1] < 2:
            messagebox.showerror("Error", "El dataset debe tener al menos 2 columnas (entradas + salida).", parent=self.win)
            return

        self.dataset = df
        self.X = df.iloc[:, :-1].to_numpy(dtype=float)
        self.Yd = df.iloc[:, -1].to_numpy(dtype=float)
        messagebox.showinfo("Dataset cargado", f"Dataset cargado: {len(df)} filas.", parent=self.win)

    # ---------- Inferir sigma (interno) ----------
    def inferir_sigma(self):
        D = self.distancias
        FA = self.fa_train
        if D is None or FA is None:
            return None
        mask = (FA > 0.0) & (FA < 1.0) & (D > 0.0)
        if not np.any(mask):
            return None
        try:
            vals = - (D[mask]**2) / (2.0 * np.log(FA[mask]))
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                return None
            sigma = float(np.median(np.sqrt(vals)))
            return sigma
        except Exception:
            return None

    # ---------- Activaciones y predicción ----------
    def calcular_activaciones(self, X, centros, sigma):
        dif = X[:, np.newaxis, :] - centros[np.newaxis, :, :]
        D2 = np.sum(dif**2, axis=2)
        denom = 2.0 * (sigma**2)
        FA = np.exp(-D2 / denom)
        return FA

    # ---------- Ejecutar simulación (sin pedir sigma) ----------
    def ejecutar_simulacion(self):
        """
        Modo automático:
        - Si JSON trae FA con filas iguales a la cantidad de patrones del dataset -> usar FA guardada.
        - Si no, inferir sigma de forma robusta (internamente) y calcular FA_test automáticamente.
        - Calcular YR = FA · W (manejo de bias/multi-salida) y después EL, EG, convergencia.
        """
        if self.modelo is None:
            messagebox.showwarning("Modelo faltante", "Primero cargue el archivo JSON del entrenamiento.", parent=self.win)
            return
        if self.dataset is None:
            messagebox.showwarning("Dataset faltante", "Primero cargue el dataset para simular.", parent=self.win)
            return

        n_patterns = self.X.shape[0]
        FA_test = None

        # 1) Usar FA guardada si tiene las mismas filas
        if self.fa_train is not None:
            try:
                fa_arr = np.asarray(self.fa_train, dtype=float)
                if fa_arr.ndim == 2 and fa_arr.shape[0] == n_patterns:
                    FA_test = fa_arr
            except Exception:
                FA_test = None

        # 2) Si no hay FA usable, inferir sigma internamente y calcular FA_test
        if FA_test is None:
            # inferir sigma robusto (intento con distancias+FA, luego heurística NN, fallback 1.0)
            def inferir_sigma_robusto(centros, distancias=None, fa=None):
                if distancias is not None and fa is not None:
                    try:
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
                # heurística nearest-neighbor sobre centros
                if centros is not None and centros.ndim == 2 and centros.shape[0] > 1:
                    try:
                        C = np.asarray(centros, dtype=float)
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

            sigma = inferir_sigma_robusto(self.centros, self.distancias, self.fa_train)
            self.sigma = sigma

            # calcular FA_test
            try:
                if self.centros is None:
                    messagebox.showerror("Sin centros", "No se encontraron centros radiales en el JSON; no puedo calcular FA para nuevos patrones.", parent=self.win)
                    return
                FA_test = self.calcular_activaciones(self.X, self.centros, sigma)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo calcular FA para datos de prueba:\n{e}", parent=self.win)
                return

        # 3) Calcular YR usando pesos W con manejo robusto de dimensiones
        try:
            if self.pesos is None:
                messagebox.showerror("Pesos faltantes", "No se encontraron pesos en el JSON. Imposible predecir.", parent=self.win)
                return

            A = np.asarray(FA_test, dtype=float)  # shape (n_patterns, M)
            W_raw = np.atleast_1d(np.asarray(self.pesos, dtype=float))
            # Caso multi-dimensional: si viene como 2D, mantener esa estructura
            if W_raw.ndim == 2:
                # si forma (M, K) o (K, M) --> tratar de inferir
                if W_raw.shape[0] == A.shape[1]:
                    W_mat = W_raw
                    out = A.dot(W_mat)  # (n_patterns, K)
                elif W_raw.shape[1] == A.shape[1]:
                    W_mat = W_raw.T
                    out = A.dot(W_mat)
                else:
                    raise ValueError(f"Pesos 2D con shape {W_raw.shape} no alinean con A cols {A.shape[1]}")
            else:
                W = W_raw.reshape(-1)  # 1D
                # caso 1: directamente alinean
                if W.size == A.shape[1]:
                    out = A.dot(W)
                # caso 2: bias incluido como último elemento
                elif W.size == A.shape[1] + 1:
                    bias = W[-1]
                    w_use = W[:-1]
                    out = A.dot(w_use) + bias
                # caso 3: W puede representar múltiples salidas concatenadas (reshape posible)
                elif W.size % A.shape[1] == 0:
                    K = W.size // A.shape[1]
                    W_mat = W.reshape(A.shape[1], K)
                    out = A.dot(W_mat)  # (n_patterns, K)
                else:
                    # no podemos alinear de forma automática -> informar claramente
                    raise ValueError(f"Dimensiones incompatibles: A cols = {A.shape[1]} , len(W) = {W.size}. "
                                     "Si el vector de pesos incluye bias, debería tener len = A.cols + 1. "
                                     "Si es multi-salida, len(W) debe ser divisible por A.cols.")
            # normalizar salida en formato YR
            if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] == 1:
                YR = out.ravel()
            else:
                YR = out  # puede ser 1D o 2D (multi-salida)
        except Exception as e:
            messagebox.showerror("Error predicción", f"No se pudo calcular YR:\n{e}", parent=self.win)
            return

        # 4) Calcular EL, EG
        try:
            YD = np.asarray(self.Yd, dtype=float).ravel()
        except Exception:
            YD = np.asarray(self.Yd)

        # Si YR es multi-columna (multi-salida), intentar comparar con YD como etiquetas (argmax)
        if isinstance(YR, np.ndarray) and YR.ndim == 2:
            # intentar inferir que YD son etiquetas enteras
            try:
                if YD.dtype.kind in 'iu' or np.all([float(x).is_integer() for x in YD]):
                    preds_idx = np.argmax(YR, axis=1)
                    EL = (YD.astype(int) - preds_idx).astype(float)
                    absEL = np.abs(EL)
                else:
                    messagebox.showerror("Multi-salida", "YR es multi-salida y YD no parece etiquetas enteras. No sé cómo comparar automáticamente.", parent=self.win)
                    return
            except Exception:
                messagebox.showerror("Multi-salida", "Error al interpretar salida multi-columna.", parent=self.win)
                return
        else:
            YR = np.asarray(YR, dtype=float).ravel()
            n = min(YR.shape[0], YD.shape[0])
            YR = YR[:n]; YD = YD[:n]
            EL = YD - YR
            absEL = np.abs(EL)

        EG = float(np.sum(absEL) / len(absEL)) if len(absEL) > 0 else float('nan')

        # 5) Comprobar epsilon
        epsilon = None
        if isinstance(self.resumen, dict):
            epsilon = self.resumen.get("error_optimo")
            try:
                epsilon = float(epsilon) if epsilon is not None else None
            except Exception:
                epsilon = None
        converge = (epsilon is not None) and (EG <= epsilon)

        # 6) Guardar y mostrar resultados
        self.ultimo_resultado = {"YR": YR, "YD": YD, "EL": EL, "absEL": absEL, "EG": EG, "sigma": self.sigma, "converge": converge, "epsilon": epsilon}
        self._mostrar_resultados_text()
        self._graficar(YD, YR, EL, EG, epsilon)
        return self.ultimo_resultado

    def _mostrar_resultados_text(self):
        R = self.ultimo_resultado
        YR, YD, EL = R["YR"], R["YD"], R["EL"]
        EG = R["EG"]
        eps = R["epsilon"]
        converge = R["converge"]

        self.txt.configure(state="normal")
        self.txt.delete("1.0", tk.END)
        # Si YR es matriz multi-salida, mostrar shape
        if isinstance(YR, np.ndarray) and YR.ndim == 2:
            self.txt.insert(tk.END, f"YR multi-salida shape: {YR.shape}\n")
        else:
            self.txt.insert(tk.END, f"Patrones: {len(YR)}\n")
        self.txt.insert(tk.END, f"EG = {EG:.6g}\n")
        if eps is not None:
            self.txt.insert(tk.END, f"Epsilon (guardado) = {eps}\n")
            self.txt.insert(tk.END, f"¿EG <= epsilon? => {'SI (CONVERGE)' if converge else 'NO'}\n")
        else:
            self.txt.insert(tk.END, "Epsilon no encontrado en el JSON\n")
        self.txt.insert(tk.END, "\nPrimeros 50 patrones (YR | YD | EL):\n")
        # Mostrar según formato YR
        if isinstance(YR, np.ndarray) and YR.ndim == 2:
            for i in range(min(50, YR.shape[0])):
                self.txt.insert(tk.END, f"{i+1}: {np.array2string(YR[i], precision=6, separator=', ')} | {YD[i]:.6g} | {EL[i]:.6g}\n")
        else:
            for i in range(min(50, len(YR))):
                self.txt.insert(tk.END, f"{i+1}: {YR[i]:.6g} | {YD[i]:.6g} | {EL[i]:.6g}\n")
        self.txt.configure(state="disabled")

    def _graficar(self, YD, YR, EL, EG, eps):
        try:
            fig, axs = plt.subplots(2,1, figsize=(8,10))
            # si YR multi-salida, graficar argmax vs YD
            if isinstance(YR, np.ndarray) and YR.ndim == 2:
                preds_idx = np.argmax(YR, axis=1)
                axs[0].scatter(YD, preds_idx, s=18)
                axs[0].set_xlabel("YD (deseada)")
                axs[0].set_ylabel("YR argmax (red)")
            else:
                axs[0].scatter(YD, YR, s=18)
                mn = min(np.min(YD), np.min(YR))
                mx = max(np.max(YD), np.max(YR))
                axs[0].plot([mn,mx],[mn,mx], linestyle="--")
                axs[0].set_xlabel("YD (deseada)")
                axs[0].set_ylabel("YR (red)")
            axs[0].set_title("YD vs YR")

            axs[1].plot(np.arange(1, len(EL)+1), EL, marker='o')
            axs[1].axhline(0, linestyle="--", linewidth=0.8)
            axs[1].set_xlabel("Patrón")
            axs[1].set_ylabel("EL = YD - YR")
            axs[1].set_title("Errores lineales")
            axs[1].text(0.02, 0.95, f"EG = {EG:.6g}", transform=axs[1].transAxes, fontsize=10, verticalalignment='top')
            if eps is not None:
                axs[1].axhline(eps, color='r', linestyle=':', linewidth=0.8)
                axs[1].axhline(-eps, color='r', linestyle=':', linewidth=0.8)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showwarning("Gráficas", f"No se pudieron generar las gráficas:\n{e}", parent=self.win)

# Función de conveniencia para llamar desde app.py
def launch_simulation_window(master):
    SimuladorRBF(master)
