# src/app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import numpy as np

# Importamos la lógica de entrenamiento RBF
from entrenamientoRBF import EntrenamientoRBF
from interpolacionRBF import InterpolacionRBF
from guardar_resultados import GuardarResultadosRBF
from preprocesamiento import Preprocesador
from simulacion_rbf import launch_simulation_window



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
        self.guardar_resultados = GuardarResultadosRBF()
        self.preprocesador = Preprocesador()

        # --------------------------------------
        # Barra superior
        # --------------------------------------
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.bt_preprocesamiento = tk.Button(
            top_frame, text="PREPROCESAMIENTO", width=18, height=2,
            command=self.show_preprocesamiento_view
        )
        self.bt_preprocesamiento.pack(side=tk.LEFT, padx=10)


        self.bt_entrenamiento = tk.Button(
            top_frame, text="ENTRENAMIENTO", width=18, height=2,
            command=self.show_entrenamiento_view
        )
        self.bt_entrenamiento.pack(side=tk.LEFT, padx=10)

        self.bt_simulacion = tk.Button(
            top_frame, text="SIMULACIÓN", width=18, height=2,
            command=lambda: launch_simulation_window(self)
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

        tk.Button(
            error_frame, text="CALCULAR ERROR GENERAL",
            command=self.calcular_interpolacion_y_pesos
        ).pack(anchor="w", pady=6)

        tk.Button(
            error_frame, text="Guardar Entrenamiento en JSON",
            command=self.guardar_entrenamiento_json
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
        # X_train: (N_patrones, n_inputs)
        X_train = self.current_train.iloc[:, :n_inputs].to_numpy(dtype=float)

        # calcular mínimo y máximo global de TODO el conjunto de entradas
        global_min = float(np.min(X_train))
        global_max = float(np.max(X_train))

        # si por alguna razón todos los valores son iguales, generamos centros idénticos
        if global_min == global_max:
            centros = np.full((self.n_centros, n_inputs), global_min, dtype=float)
        else:
            centros = np.random.uniform(low=global_min, high=global_max, size=(self.n_centros, n_inputs))

        # Guardar e imprimir
        self.centros_radiales = centros
        self.entrenamiento_rbf.centros_radiales = centros

        self.txt_centros.configure(state="normal")
        self.txt_centros.delete("1.0", tk.END)
        self.txt_centros.insert(tk.END, f"Matriz de centros radiales ({self.n_centros} x {n_inputs}):\n\n")
        self.txt_centros.insert(tk.END, np.array2string(centros, precision=4, suppress_small=True))
        self.txt_centros.configure(state="disabled")

        messagebox.showinfo("Centros inicializados", f"Centros radiales generados correctamente.\nRango global usado: [{global_min:.6g}, {global_max:.6g}]")


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

    def guardar_entrenamiento_json(self):
        """Guarda toda la información del entrenamiento en un archivo JSON."""
        if self.summary is None:
            messagebox.showwarning("Sin datos", "Debe cargar un dataset antes de guardar.")
            return

        resumen = {
            "entradas": self.summary.get("entradas"),
            "salidas": self.summary.get("salidas"),
            "patrones": self.summary.get("patrones"),
            "error_optimo": self.entrenamiento_rbf.error_optimo,
            "num_centros": self.n_centros
        }

        centros = self.centros_radiales
        distancias = self.entrenamiento_rbf.distancias
        fa = self.entrenamiento_rbf.funcion_activacion
        matriz_interp = self.interpolacion_rbf.matriz_A if hasattr(self.interpolacion_rbf, "matriz_A") else None
        pesos = self.interpolacion_rbf.pesos if hasattr(self.interpolacion_rbf, "pesos") else None

        self.guardar_resultados.guardar(resumen, centros, distancias, fa, matriz_interp, pesos)

    # ==========================================
    # Vista: PREPROCESAMIENTO
    # ==========================================
    def show_preprocesamiento_view(self):
        self.clear_content()
        frame = tk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(left, text="Seleccionar dataset para preprocesar:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.prep_dataset_combobox = ttk.Combobox(left, state="readonly", width=60)
        self.prep_dataset_combobox.pack(anchor="w", pady=5)
        files = self.scan_datasets()
        self.prep_dataset_combobox["values"] = files
        self.prep_dataset_combobox.bind("<<ComboboxSelected>>", self.on_preproc_dataset_selected)

        tk.Button(left, text="Cargar otro archivo...", command=lambda: self.load_other_file()).pack(anchor="w", pady=6)

        # Opciones de preprocesamiento
        opts_frame = tk.LabelFrame(left, text="Opciones de preprocesamiento", padx=6, pady=6)
        opts_frame.pack(fill=tk.X, pady=6)

        tk.Label(opts_frame, text="Método de relleno (numérico):").pack(anchor="w")
        self.combo_fill = ttk.Combobox(opts_frame, values=["mean", "median", "mode", "ffill", "bfill"], state="readonly", width=12)
        self.combo_fill.set("mean")
        self.combo_fill.pack(anchor="w", pady=4)

        tk.Label(opts_frame, text="Relleno categórico:").pack(anchor="w")
        self.combo_fill_cat = ttk.Combobox(opts_frame, values=["mode", "unknown"], state="readonly", width=12)
        self.combo_fill_cat.set("mode")
        self.combo_fill_cat.pack(anchor="w", pady=4)

        tk.Label(opts_frame, text="Eliminar columnas con > % faltantes:").pack(anchor="w")
        self.entry_drop_thr = tk.Entry(opts_frame, width=6)
        self.entry_drop_thr.insert(0, "0.5")
        self.entry_drop_thr.pack(anchor="w", pady=4)

        tk.Label(opts_frame, text="Escalado X:").pack(anchor="w")
        self.combo_scale = ttk.Combobox(opts_frame, values=["None", "standard", "minmax"], state="readonly", width=12)
        self.combo_scale.set("None")
        self.combo_scale.pack(anchor="w", pady=4)

        tk.Label(opts_frame, text="Columna Y (dejar vacío → última columna):").pack(anchor="w")
        self.entry_ycol = tk.Entry(opts_frame, width=20)
        self.entry_ycol.pack(anchor="w", pady=4)

        tk.Button(opts_frame, text="Ejecutar Preprocesamiento", command=self.run_preprocessing).pack(anchor="w", pady=8)

        # Preview / resultado
        right = tk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(right, text="Vista previa procesada (primeras 10 filas):", font=("Arial", 11, "bold")).pack(anchor="w", pady=6)
        self.txt_prep_preview = tk.Text(right, height=30, width=90)
        self.txt_prep_preview.pack(fill=tk.BOTH, expand=True)
        self.txt_prep_preview.configure(state="disabled")

        tk.Button(right, text="Guardar dataset preprocesado", command=self.save_preprocessed_file).pack(anchor="e", pady=8)

        # Variables de estado
        self.preprocessed_df = None
        self.preproc_report = None

    def on_preproc_dataset_selected(self, event=None):
        sel = self.prep_dataset_combobox.get()
        if not sel:
            return
        # Reusar tu load_dataset
        try:
            df = self.load_dataset(sel)
        except Exception as e:
            messagebox.showerror("Error al cargar", f"No se pudo leer el archivo:\n{e}")
            return

        # mostrar pequeña preview original
        self.txt_prep_preview.configure(state="normal")
        self.txt_prep_preview.delete("1.0", tk.END)
        self.txt_prep_preview.insert(tk.END, "Preview original (30 primeras filas):\n")
        self.txt_prep_preview.insert(tk.END, df.head(30).to_string(index=False))
        self.txt_prep_preview.configure(state="disabled")

        # guardar ruta actual para que run_preprocessing la use
        self.preproc_current_path = sel
        self.preproc_original_df = df

    def run_preprocessing(self):
        # Validaciones
        if not hasattr(self, "preproc_original_df") or self.preproc_original_df is None:
            messagebox.showwarning("Sin dataset", "Seleccione un dataset para preprocesar.")
            return

        try:
            drop_thr = float(self.entry_drop_thr.get())
        except Exception:
            messagebox.showerror("Valor inválido", "El umbral de eliminación debe ser un número entre 0 y 1.")
            return

        fill_strategy = self.combo_fill.get()
        fill_cat = self.combo_fill_cat.get()
        scale_method = self.combo_scale.get()
        scale_method = None if scale_method == "None" else scale_method
        ycol = self.entry_ycol.get().strip() or None

        # ejecutar pipeline
        df_in = self.preproc_original_df.copy()
        processed_df, report = self.preprocesador.process(
            df=df_in,
            original_path=getattr(self, "preproc_current_path", None),
            y_column=ycol,
            fill_strategy=fill_strategy,
            fill_categorical=fill_cat,
            drop_threshold=drop_thr,
            drop_rows=False,
            scale_method=scale_method
        )

        self.preprocessed_df = processed_df
        self.preproc_report = report

        # mostrar preview
        self.txt_prep_preview.configure(state="normal")
        self.txt_prep_preview.delete("1.0", tk.END)
        self.txt_prep_preview.insert(tk.END, processed_df.head(30).to_string(index=False))
        self.txt_prep_preview.insert(tk.END, "\n\nReporte (extracto):\n")
        # mostrar resumen corto del reporte
        lines = []
        gd = report.get("global", {})
        lines.append(f"Y column: {gd.get('y_column')}")
        lines.append(f"Filas después: {gd.get('rows_after')}")
        lines.append(f"Columnas después: {', '.join(gd.get('columns_after', []))}")
        dropped_cols = gd.get("dropped_columns_high_missing", [])
        if dropped_cols:
            lines.append(f"Columnas eliminadas (missing alto): {', '.join(dropped_cols)}")
        self.txt_prep_preview.insert(tk.END, "\n".join(lines))
        self.txt_prep_preview.configure(state="disabled")

        messagebox.showinfo("Preprocesamiento", "Preprocesamiento completado correctamente. Revise la vista previa y guarde el dataset si lo desea.")

    def save_preprocessed_file(self):
        if self.preprocessed_df is None:
            messagebox.showwarning("Nada para guardar", "No hay dataset preprocesado. Ejecute el preprocesamiento primero.")
            return

        orig = getattr(self, "preproc_current_path", None)
        # preguntar nombre / carpeta con filedialog
        if orig:
            orig_path = Path(orig)
            default_name = orig_path.stem + "_preprocessed" + orig_path.suffix
            initial_dir = str(orig_path.parent)
            filetypes = [("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")]
        else:
            default_name = "preprocessed_dataset.csv"
            initial_dir = "."
            filetypes = [("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")]

        save_path = filedialog.asksaveasfilename(
            title="Guardar dataset preprocesado",
            initialdir=initial_dir,
            initialfile=default_name,
            filetypes=filetypes,
            defaultextension=Path(default_name).suffix
        )
        if not save_path:
            return

        # usar preprocesador.save_processed para guardar y el reporte
        out_path, report_path = self.preprocesador.save_processed(
            self.preprocessed_df,
            original_path=getattr(self, "preproc_current_path", None),
            target_path=save_path,
            report=self.preproc_report
        )

        messagebox.showinfo("Guardado", f"Dataset guardado en:\n{out_path}\n\nReporte guardado en:\n{report_path}")



# ==========================================
# Punto de entrada
# ==========================================
if __name__ == "__main__":
    app = RBFApp()
    app.mainloop()
