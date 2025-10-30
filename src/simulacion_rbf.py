import json
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np


def launch_simulation_panel(parent_container, app_ref=None):
    for child in parent_container.winfo_children():
        child.destroy()
    panel = SimulacionPanel(parent_container, app_ref)
    panel.pack(fill=tk.BOTH, expand=True)
    return panel


def _pesos_from_json(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        try:
            items = sorted(obj.items(), key=lambda kv: int(''.join([c for c in kv[0] if c.isdigit()]) or -1))
        except Exception:
            items = sorted(obj.items())
        return np.array([float(v) for k, v in items], dtype=float)
    return np.array(obj, dtype=float)


class SimulacionPanel(tk.Frame):
    def __init__(self, parent, app_ref=None):
        super().__init__(parent)
        self.app_ref = app_ref
        self.pack(fill=tk.BOTH, expand=True)

        tk.Label(self, text='Simulación RBF — Cargar JSON', font=('Arial', 12, 'bold')).pack(pady=8)
        tk.Button(self, text='Cargar JSON', command=self.load_json).pack(padx=10, pady=5)

        self.inputs_frame = tk.Frame(self)
        self.inputs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.entries = {}  # guardar entradas dinámicas
        # datos de la red cargados desde el JSON
        self.centros = None  # shape: (n_centros, n_entradas)
        self.W = None        # shape: (n_centros+1,)
        self.errorMSE = None
        self.error_permitido = None
        self.codificaciones = {}
        self.columnas = []
        self.input_cols = []
        self.output_cols = []

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Error leyendo JSON:{e}')
            return

        resumen = data.get('resumen', {}) if isinstance(data, dict) else {}
        # PRECEDENCIA: input_names explícito (lo que pediste)
        input_cols = data.get('input_names') or resumen.get('input_names')
        # outputs explícitos si existen
        output_cols = data.get('output_names') or resumen.get('output_names') or data.get('output_name') or resumen.get('output_name')

        # si no hay input_names, intentar columnas o generar por entradas
        columns = resumen.get('columns') or data.get('columns')
        entradas = resumen.get('entradas') or data.get('entradas')

        if not input_cols:
            if isinstance(columns, list) and columns:
                # asumimos que todas menos la última son inputs si no hay input_names
                if len(columns) > 1:
                    input_cols = columns[:-1]
                    if not output_cols:
                        output_cols = [columns[-1]]
                else:
                    # una sola columna -> tratar como input
                    input_cols = columns
            elif entradas:
                try:
                    n = int(entradas)
                    input_cols = [f'X{i+1}' for i in range(n)]
                except Exception:
                    input_cols = []
            else:
                input_cols = []

        # si no hay outputs, intentar inferir
        if not output_cols and isinstance(columns, list) and columns:
            # si input_cols ya coincide con parte de columns, la salida será lo restante
            if set(input_cols) and all(c in columns for c in input_cols):
                output_cols = [c for c in columns if c not in input_cols]
            if not output_cols and len(columns) > len(input_cols):
                output_cols = [columns[-1]]

        # cargar centros radiales y pesos desde el JSON si existen
        centros_json = data.get('centros_radiales') or data.get('centros') or data.get('radios') or data.get('radios_centros')
        if centros_json:
            try:
                self.centros = np.array(centros_json, dtype=float)
            except Exception:
                self.centros = None

        pesos_json = data.get('pesos')
        self.W = _pesos_from_json(pesos_json)

        # errores y codificaciones opcionales
        self.errorMSE = data.get('errorMSE') or resumen.get('errorMSE') or resumen.get('error_optimo')
        self.error_permitido = resumen.get('error_optimo') or data.get('error_optimo')
        self.codificaciones = data.get('codificaciones') or data.get('mapeos') or {}
        self.columnas = columns or []
        self.input_cols = input_cols or []
        self.output_cols = output_cols or []

        # limpiar contenido anterior
        for w in self.inputs_frame.winfo_children():
            w.destroy()
        self.entries.clear()

        if not self.input_cols:
            tk.Label(self.inputs_frame, text='No se encontraron nombres de entradas (input_names) en el JSON.').pack()
            return

        # crear labels + entries dinámicos SOLO para inputs
        for i, col in enumerate(self.input_cols, 1):
            frame = tk.Frame(self.inputs_frame)
            frame.pack(fill=tk.X, pady=3)
            tk.Label(frame, text=f'{col}:', width=25, anchor='w').pack(side=tk.LEFT)
            var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=var, width=20)
            entry.pack(side=tk.LEFT)
            self.entries[col] = var

        info_text = 'Entradas cargadas: ' + ', '.join(self.input_cols)
        if self.output_cols:
            info_text += 'Salida(s) detectada(s): ' + ', '.join(self.output_cols)
        tk.Label(self.inputs_frame, text=info_text).pack(pady=6)

        tk.Button(self.inputs_frame, text='CALCULAR SALIDA', command=self.calcular_salida).pack(pady=10)

    def calcular_salida(self):
        # validaciones simples
        if self.W is None or self.centros is None:
            messagebox.showwarning('Aviso', 'No hay pesos o centros cargados en el JSON.')
            return

        try:
            # capturar entradas en el orden de self.input_cols
            entradas_usuario = []
            for col in self.input_cols:
                s = self.entries[col].get()
                if s == '':
                    raise ValueError(f'La entrada "{col}" está vacía.')
                entradas_usuario.append(float(s))

            x = np.array(entradas_usuario, dtype=float).reshape(-1)  # vector (n_entradas,)

            # calcular distancias a cada centro R_j
            if x.shape[0] != self.centros.shape[1]:
                raise ValueError(f'Número de entradas ({x.shape[0]}) no coincide con centros (esperan {self.centros.shape[1]}).')

            D = np.linalg.norm(self.centros - x, axis=1)  # shape (n_centros,)
            D_safe = np.where(D <= 0, 1e-6, D)
            FA = (D_safe ** 2) * np.log(D_safe)  # función de activación para cada centro

            # construir vector phi: [1, FA1, FA2, ...]
            phi = np.concatenate(([1.0], FA))  # shape (n_centros+1,)

            # asegurar que W tenga longitud compatible
            W = self.W.flatten()
            if W.shape[0] != phi.shape[0]:
                if W.shape[0] > phi.shape[0]:
                    W = W[:phi.shape[0]]
                else:
                    raise ValueError(f'Longitud de pesos ({W.shape[0]}) no coincide con phi ({phi.shape[0]}).')

            salida_valor = float(np.dot(phi, W))

            # tratar codificaciones (si la última columna es categórica mapeada)
            salida_nombre = self.output_cols[0] if self.output_cols else 'Salida'
            codif = self.codificaciones.get(salida_nombre, {}) if isinstance(self.codificaciones, dict) else {}

            if isinstance(codif, dict) and codif:
                inverso = {float(v): k for k, v in codif.items()}
                codigo_closer = min(inverso.keys(), key=lambda c: abs(c - salida_valor))
                categoria = inverso[codigo_closer]
                mensaje = f'Salida estimada ({salida_nombre}): {salida_valor:.6f} Categoría: {categoria}'
            else:
                mensaje = f'Salida estimada ({salida_nombre}): {salida_valor:.6f}'

            messagebox.showinfo('Resultado de simulación', mensaje)

        except Exception as e:
            messagebox.showerror('Error en simulación', str(e))


# Ejemplo de uso desde la ventana principal:
# def show_simulacion_view(self):
#     launch_simulation_panel(self.content, self)
