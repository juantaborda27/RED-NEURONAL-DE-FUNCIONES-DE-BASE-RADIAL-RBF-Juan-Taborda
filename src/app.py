# src/app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import numpy as np

# Importamos la lógica de entrenamiento RBF
from entrenamientoRBF import EntrenamientoRBF
from interpolacionRBF import InterpolacionRBF

DATASETS_FOLDER = Path("datasets")


class RBFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Red Neuronal de Funciones de Base Radial (RBF)")
        self.geometry("1800x950")
        self.resizable(True, True)

        # Estado actual
        self.current_train = None
        self.current_test = None
        self.current_path = None
        self.summary = None
        self.n_centros = None
        self.centros_radiales = None

        # Instancia del módulo de entrenamiento
        self.entrenamiento_rbf = EntrenamientoRBF()
        self.interpolacion_rbf = InterpolacionRBF()

        # --------------------------------------
        # Barra superior
        # --------------------------------------
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.bt_entrenamiento = tk.Button(
            top_frame, text="ENTRENAMIENTO", width=18, height=2,
            command=self.show_entrenamiento_view
        )
        self.bt_entrenamiento.pack(side=tk.LEFT, padx=10)

        self.bt_simulacion = tk.Button(
            top_frame, text="SIMULACIÓN", width=18, height=2,
            command=self.show_simulacion_view
        )
        self.bt_simulacion.pack(side=tk.LEFT, padx=10)

        # Contenedor dinámico
        self.content = tk.Frame(self)
        self.content.pack(fill=tk.BOTH, expand=True)

        # Mostrar por defecto vista de entrenamiento
        self.show_entrenamiento_view()

    # ==========================================
    # Utilidades
    # ==========================================
    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def scan_datasets(self):
        if not DATASETS_FOLDER.exists():
            return []
        return [str(f) for f in sorted(DATASETS_FOLDER.iterdir())
                if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]

    def load_dataset(self, path: str) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        else:
            raise ValueError("Formato no soportado. Usa CSV o Excel.")

    # ==========================================
    # Vista: ENTRENAMIENTO
    # ==========================================
    def show_entrenamiento_view(self):
        self.clear_content()
        frame = tk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel izquierdo
        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(left, text="Seleccionar dataset:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.dataset_combobox = ttk.Combobox(left, state="readonly", width=60)
        self.dataset_combobox.pack(anchor="w", pady=5)

        files = self.scan_datasets()
        self.dataset_combobox["values"] = files
        # No carga automática
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.on_dataset_selected)

        tk.Button(left, text="Cargar otro archivo...", command=self.load_other_file).pack(anchor="w", pady=6)

        # Resumen del dataset
        info_frame = tk.LabelFrame(left, text="Resumen del dataset", padx=6, pady=6)
        info_frame.pack(fill=tk.X, pady=6)
        self.lbl_patrones = tk.Label(info_frame, text="Patrones: -")
        self.lbl_patrones.pack(anchor="w")
        self.lbl_entradas = tk.Label(info_frame, text="Entradas: -")
        self.lbl_entradas.pack(anchor="w")
        self.lbl_salidas = tk.Label(info_frame, text="Salidas: -")
        self.lbl_salidas.pack(anchor="w")
        self.lbl_cols = tk.Label(info_frame, text="Columnas: -", wraplength=320, justify="left")
        self.lbl_cols.pack(anchor="w", pady=(4, 0))

        # Configuración RBF
        centros_frame = tk.LabelFrame(left, text="Configuración RBF", padx=6, pady=6)
        centros_frame.pack(fill=tk.X, pady=6)
        tk.Label(centros_frame, text="Número de centros radiales:").pack(anchor="w")
        self.entry_centros = tk.Entry(centros_frame, width=10)
        self.entry_centros.pack(anchor="w", pady=4)
        tk.Button(centros_frame, text="Guardar número de centros", command=self.set_num_centros).pack(anchor="w", pady=4)

        # Vista previa
        preview_frame = tk.LabelFrame(left, text="Vista previa (5 primeros)", padx=6, pady=6)
        preview_frame.pack(fill=tk.BOTH, expand=False, pady=6)
        self.txt_preview = tk.Text(preview_frame, height=10, width=70)
        self.txt_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt_preview.configure(state="disabled")

        # Panel derecho
        right = tk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(right, text="Inicialización de Centros Radiales", font=("Arial", 12, "bold")).pack(anchor="w", pady=6)
        tk.Button(
            right, text="Inicializar centros radiales", width=25, height=2,
            command=self.inicializar_centros
        ).pack(anchor="w", pady=6)

        # Mostrar centros
        self.txt_centros = tk.Text(right, height=12, width=80)
        self.txt_centros.pack(fill=tk.BOTH, expand=True, pady=6)
        self.txt_centros.configure(state="disabled")

        # Error óptimo
        error_frame = tk.LabelFrame(right, text="Error de aproximación óptimo (ε)", padx=6, pady=6)
        error_frame.pack(fill=tk.X, pady=6)
        tk.Label(error_frame, text="Ingrese ε (0 < ε ≤ 0.1):").pack(anchor="w")
        self.entry_error_optimo = tk.Entry(error_frame, width=10)
        self.entry_error_optimo.insert(0, "0.1")
        self.entry_error_optimo.pack(anchor="w", pady=4)

        tk.Button(
            error_frame, text="Calcular Distancias y FA",
            command=self.calcular_distancias_y_fa
        ).pack(anchor="w", pady=6)

        tk.Button(
            error_frame, text="Calcular Matriz de Interpolación y Pesos",
            command=self.calcular_interpolacion_y_pesos
        ).pack(anchor="w", pady=6)


        # ================================
        # Panel dividido para los resultados
        # ================================
        result_container = tk.Frame(right)
        result_container.pack(fill=tk.BOTH, expand=True, pady=8)

        # Frame izquierdo: Distancias y FA
        frame_dist = tk.LabelFrame(result_container, text="Resultados: Distancias y FA", padx=6, pady=6)
        frame_dist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.txt_resultados_dist = tk.Text(frame_dist, height=20, wrap="word")
        self.txt_resultados_dist.pack(fill=tk.BOTH, expand=True)
        self.txt_resultados_dist.configure(state="disabled")

        # Frame derecho: Matriz de Interpolación y Pesos
        frame_interp = tk.LabelFrame(result_container, text="Matriz de Interpolación y Pesos", padx=6, pady=6)
        frame_interp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.txt_resultados_interp = tk.Text(frame_interp, height=20, wrap="word")
        self.txt_resultados_interp.pack(fill=tk.BOTH, expand=True)
        self.txt_resultados_interp.configure(state="disabled")


    # ==========================================
    # Vista: SIMULACIÓN
    # ==========================================
    def show_simulacion_view(self):
        self.clear_content()
        tk.Label(
            self.content,
            text="Vista de simulación RBF (en desarrollo)",
            font=("Arial", 14, "bold")
        ).pack(pady=50)

    # ==========================================
    # Acciones
    # ==========================================
    def load_other_file(self):
        file = filedialog.askopenfilename(
            title="Seleccionar dataset",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")]
        )
        if file:
            vals = list(self.dataset_combobox["values"])
            if file not in vals:
                vals.append(file)
                self.dataset_combobox["values"] = vals
            idx = self.dataset_combobox["values"].index(file)
            self.dataset_combobox.current(idx)
            self.on_dataset_selected()

    def on_dataset_selected(self, event=None):
        sel = self.dataset_combobox.get()
        if not sel:
            return
        try:
            df = self.load_dataset(sel)
        except Exception as e:
            messagebox.showerror("Error al cargar", f"No se pudo leer el archivo:\n{e}")
            return

        total_patrones = len(df)
        train_size = int(0.7 * total_patrones)

        # ✅ 70% inicial → entrenamiento | 30% final → simulación
        df_train = df.iloc[:train_size].copy()   # primeras filas (70%)
        df_test = df.iloc[train_size:].copy()    # últimas filas (30%)

        self.current_train = df_train.reset_index(drop=True)
        self.current_test = df_test.reset_index(drop=True)
        self.current_path = sel

        n_cols = df.shape[1]
        n_inputs = n_cols - 1
        n_outputs = 1

        self.summary = {
            "entradas": n_inputs,
            "salidas": n_outputs,
            "patrones": total_patrones,
            "columns": list(df.columns)
        }

        self.lbl_patrones.config(text=f"Patrones totales: {total_patrones}")
        self.lbl_entradas.config(text=f"Entradas: {n_inputs}")
        self.lbl_salidas.config(text=f"Salidas: {n_outputs}")
        self.lbl_cols.config(text=f"Columnas: {', '.join(list(df.columns))}")

        self.txt_preview.configure(state="normal")
        self.txt_preview.delete("1.0", tk.END)
        self.txt_preview.insert(tk.END, df.head(5).to_string(index=False))
        self.txt_preview.configure(state="disabled")

        messagebox.showinfo(
            "Partición completada",
            f"Dataset '{Path(sel).name}' dividido automáticamente:\n"
            f"→ 70% (primeros valores) para entrenamiento ({len(df_train)})\n"
            f"→ 30% (últimos valores) para simulación ({len(df_test)})"
        )

    def set_num_centros(self):
        if not self.summary:
            messagebox.showwarning("Dataset no seleccionado", "Debe seleccionar un dataset primero.")
            return

        entradas = self.summary["entradas"]
        val = self.entry_centros.get().strip()
        if not val.isdigit():
            messagebox.showerror("Valor inválido", "Ingrese un número entero para los centros radiales.")
            return

        n_centros = int(val)
        if n_centros < entradas:
            messagebox.showerror("Valor inválido",
                                 f"El número de centros ({n_centros}) debe ser ≥ número de entradas ({entradas}).")
            return

        self.n_centros = n_centros
        messagebox.showinfo("Número de centros", f"Centros radiales = {n_centros}")

    def inicializar_centros(self):
        if self.current_train is None or self.summary is None:
            messagebox.showwarning("Dataset no cargado", "Seleccione un dataset antes.")
            return
        if self.n_centros is None:
            messagebox.showwarning("Centros no definidos", "Ingrese y guarde el número de centros primero.")
            return

        n_inputs = self.summary["entradas"]
        X_train = self.current_train.iloc[:, :n_inputs].to_numpy(dtype=float)
        min_vals, max_vals = X_train.min(axis=0), X_train.max(axis=0)

        centros = np.random.uniform(low=min_vals, high=max_vals, size=(self.n_centros, n_inputs))
        self.centros_radiales = centros
        self.entrenamiento_rbf.centros_radiales = centros

        self.txt_centros.configure(state="normal")
        self.txt_centros.delete("1.0", tk.END)
        self.txt_centros.insert(tk.END, f"Matriz de centros radiales ({self.n_centros} x {n_inputs}):\n\n")
        self.txt_centros.insert(tk.END, np.array2string(centros, precision=4, suppress_small=True))
        self.txt_centros.configure(state="disabled")

        messagebox.showinfo("Centros inicializados", "Centros radiales generados correctamente.")

    def calcular_distancias_y_fa(self):
        if self.current_train is None or self.centros_radiales is None:
            messagebox.showwarning("Datos incompletos", "Cargue dataset e inicialice los centros radiales.")
            return

        try:
            eps = float(self.entry_error_optimo.get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido para ε.")
            return

        if not self.entrenamiento_rbf.set_error_optimo(eps):
            return

        X = self.current_train.iloc[:, :-1].to_numpy(dtype=float)
        self.entrenamiento_rbf.calcular_distancias(X, self.centros_radiales)
        self.entrenamiento_rbf.calcular_funcion_activacion()

        resumen = self.entrenamiento_rbf.generar_resumen_texto(max_rows=50)
        self.txt_resultados_dist.configure(state="normal")
        self.txt_resultados_dist.delete("1.0", tk.END)
        self.txt_resultados_dist.insert(tk.END, resumen)
        self.txt_resultados_dist.configure(state="disabled")


        messagebox.showinfo("Cálculos completados", "Distancias y FA calculadas correctamente.")

    def calcular_interpolacion_y_pesos(self):
        if self.entrenamiento_rbf.funcion_activacion is None:
            messagebox.showwarning("Faltan datos", "Debe calcular la función de activación primero.")
            return

        Y = self.current_train.iloc[:, -1].to_numpy(dtype=float)
        FA = self.entrenamiento_rbf.funcion_activacion

        self.interpolacion_rbf.calcular_pesos(FA, Y)
        resumen = self.interpolacion_rbf.generar_resumen_texto()

        self.txt_resultados_interp.configure(state="normal")
        self.txt_resultados_interp.delete("1.0", tk.END)
        self.txt_resultados_interp.insert(tk.END, resumen)
        self.txt_resultados_interp.configure(state="disabled")


        messagebox.showinfo("Matriz A y Pesos", "Cálculo de la matriz de interpolación y pesos completado.")

# ==========================================
# Punto de entrada
# ==========================================
if __name__ == "__main__":
    app = RBFApp()
    app.mainloop()
