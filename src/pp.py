import json
from tkinter import messagebox, filedialog
import numpy as np
from pathlib import Path


# ==========================================
# Vista: SIMULACIÓN
# ==========================================
def show_simulacion_view(self):
    self.clear_content()
    frame = tk.Frame(self.content)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Panel izquierdo - Carga de JSON y configuración
    left = tk.Frame(frame)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    
    tk.Label(left, text="Cargar entrenamiento desde JSON:", 
             font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
    
    tk.Button(left, text="📁 Cargar archivo JSON...", 
              command=self.cargar_json_entrenamiento).pack(anchor="w", pady=5)
    
    # Información del entrenamiento cargado
    info_frame = tk.LabelFrame(left, text="Información del entrenamiento", padx=6, pady=6)
    info_frame.pack(fill=tk.X, pady=10)
    
    self.lbl_json_info = tk.Label(info_frame, text="No se ha cargado ningún entrenamiento", 
                                 wraplength=300, justify="left")
    self.lbl_json_info.pack(anchor="w")
    
    # Configuración de simulación
    sim_frame = tk.LabelFrame(left, text="Configuración de simulación", padx=6, pady=6)
    sim_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(sim_frame, text="Error de aproximación óptimo:").pack(anchor="w")
    self.entry_error_sim = tk.Entry(sim_frame, width=10)
    self.entry_error_sim.insert(0, "0.1")
    self.entry_error_sim.pack(anchor="w", pady=4)
    
    tk.Button(sim_frame, text="Ejecutar Simulación", 
              command=self.ejecutar_simulacion).pack(anchor="w", pady=8)
    
    # Panel derecho - Resultados de simulación
    right = tk.Frame(frame)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Resultados de YR vs YD
    resultados_frame = tk.LabelFrame(right, text="Resultados de Simulación", padx=6, pady=6)
    resultados_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Crear un frame con scroll para los resultados
    result_container = tk.Frame(resultados_frame)
    result_container.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(result_container)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    self.txt_resultados_sim = tk.Text(result_container, height=20, wrap="word", 
                                     yscrollcommand=scrollbar.set)
    self.txt_resultados_sim.pack(fill=tk.BOTH, expand=True)
    self.txt_resultados_sim.configure(state="disabled")
    
    scrollbar.config(command=self.txt_resultados_sim.yview)
    
    # Resultado final de convergencia
    self.lbl_convergencia = tk.Label(right, text="", font=("Arial", 12, "bold"))
    self.lbl_convergencia.pack(pady=10)

def cargar_json_entrenamiento(self):
    """Carga un archivo JSON de entrenamiento guardado previamente."""
    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo JSON de entrenamiento",
        filetypes=[("Archivos JSON", "*.json")]
    )
    
    if not archivo:
        return
    
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            self.datos_entrenamiento = json.load(f)
        
        # Mostrar información del entrenamiento
        resumen = self.datos_entrenamiento.get('resumen', {})
        info_text = f"Archivo: {Path(archivo).name}\n"
        info_text += f"Entradas: {resumen.get('entradas', 'N/A')}\n"
        info_text += f"Salidas: {resumen.get('salidas', 'N/A')}\n"
        info_text += f"Patrones: {resumen.get('patrones', 'N/A')}\n"
        info_text += f"Centros radiales: {resumen.get('num_centros', 'N/A')}\n"
        info_text += f"Error óptimo: {resumen.get('error_optimo', 'N/A')}"
        
        self.lbl_json_info.config(text=info_text)
        messagebox.showinfo("Carga exitosa", "Entrenamiento cargado correctamente.")
        
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo JSON:\n{e}")

def ejecutar_simulacion(self):
    """Ejecuta la simulación con los datos del entrenamiento cargado."""
    if not hasattr(self, 'datos_entrenamiento'):
        messagebox.showwarning("Sin datos", "Debe cargar un archivo JSON de entrenamiento primero.")
        return
    
    try:
        # Obtener datos del entrenamiento
        centros = np.array(self.datos_entrenamiento['centros_radiales'])
        pesos_dict = self.datos_entrenamiento['pesos']
        pesos = np.array([pesos_dict[f'W{i}'] for i in range(len(pesos_dict))])
        error_optimo = float(self.entry_error_sim.get())
        
        # Cargar datos de prueba (30% final del dataset original)
        if self.current_test is None:
            messagebox.showwarning("Sin datos de prueba", 
                                 "No hay dataset de prueba cargado. Cargue un dataset primero en la vista de entrenamiento.")
            return
        
        n_inputs = self.summary["entradas"]
        X_test = self.current_test.iloc[:, :n_inputs].to_numpy(dtype=float)
        Y_test = self.current_test.iloc[:, -1].to_numpy(dtype=float)
        
        # Calcular distancias para los patrones de prueba
        n_patrones = X_test.shape[0]
        n_centros = centros.shape[0]
        distancias = np.zeros((n_patrones, n_centros))
        
        for i in range(n_patrones):
            for j in range(n_centros):
                distancias[i, j] = np.linalg.norm(X_test[i] - centros[j])
        
        # Calcular función de activación (FA)
        fa = np.exp(-distancias)
        
        # Calcular salidas de la red (YR)
        YR = np.zeros(n_patrones)
        for i in range(n_patrones):
            # Wo + sum(Wi * FAi)
            YR[i] = pesos[0]  # Wo
            for j in range(1, len(pesos)):
                YR[i] += pesos[j] * fa[i, j-1]
        
        # Calcular errores lineales (EL)
        EL = Y_test - YR
        
        # Calcular error general (EG)
        EG = np.sum(np.abs(EL)) / n_patrones
        
        # Verificar convergencia
        converge = EG <= error_optimo
        
        # Mostrar resultados
        self.mostrar_resultados_simulacion(Y_test, YR, EL, EG, converge, error_optimo)
        
    except Exception as e:
        messagebox.showerror("Error en simulación", f"Ocurrió un error durante la simulación:\n{e}")

def mostrar_resultados_simulacion(self, YD, YR, EL, EG, converge, error_optimo):
    """Muestra los resultados de la simulación en el área de texto."""
    self.txt_resultados_sim.configure(state="normal")
    self.txt_resultados_sim.delete("1.0", tk.END)
    
    # Encabezado
    self.txt_resultados_sim.insert(tk.END, "RESULTADOS DE SIMULACIÓN\n")
    self.txt_resultados_sim.insert(tk.END, "=" * 50 + "\n\n")
    
    # Mostrar cada patrón
    for i in range(len(YD)):
        self.txt_resultados_sim.insert(tk.END, f"Patrón {i+1}:\n")
        self.txt_resultados_sim.insert(tk.END, f"  YD({i+1}) = {YD[i]:.4f}\n")
        self.txt_resultados_sim.insert(tk.END, f"  YR({i+1}) = {YR[i]:.4f}\n")
        self.txt_resultados_sim.insert(tk.END, f"  EL({i+1}) = YD - YR = {EL[i]:.4f}\n")
        self.txt_resultados_sim.insert(tk.END, "-" * 30 + "\n")
    
    # Mostrar error general
    self.txt_resultados_sim.insert(tk.END, f"\nERROR GENERAL (EG) = ∑|EL| / No. Patrones\n")
    self.txt_resultados_sim.insert(tk.END, f"EG = {EG:.4f}\n\n")
    
    # Mostrar condición de parada
    self.txt_resultados_sim.insert(tk.END, f"CONDICIÓN DE PARADA:\n")
    self.txt_resultados_sim.insert(tk.END, f"EG ({EG:.4f}) <= Error de aproximación óptimo ({error_optimo:.4f})\n")
    
    self.txt_resultados_sim.configure(state="disabled")
    
    # Mostrar resultado de convergencia
    if converge:
        self.lbl_convergencia.config(
            text="✓ LA RED CONVERGE - Condición de parada cumplida", 
            fg="green"
        )
    else:
        self.lbl_convergencia.config(
            text="✗ LA RED NO CONVERGE - Condición de parada no cumplida", 
            fg="red"
        )