import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
        # --- estilos básicos ---
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Encabezado
        hdr = ttk.Frame(self)
        hdr.pack(fill=tk.X, padx=12, pady=(10, 6))
        ttk.Label(hdr, text='Simulación RBF', font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        ttk.Button(hdr, text='Cargar JSON', command=self.load_json).pack(side=tk.RIGHT)

        # Paned window para dividir izquierda (entradas) y derecha (resumen+resultado)
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Panel izquierdo: entradas y controles
        left = ttk.Frame(paned)
        paned.add(left, weight=3)

        left_top = ttk.LabelFrame(left, text="Entradas para simulación")
        left_top.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Contenedor con grid para labels+entries
        self.form_frame = ttk.Frame(left_top)
        self.form_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # area de botones en la parte inferior izquierda
        left_bottom = ttk.Frame(left)
        left_bottom.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.btn_calc = ttk.Button(left_bottom, text='CALCULAR SALIDA', command=self.calcular_salida)
        self.btn_calc.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(left_bottom, text='Limpiar', command=self._limpiar_inputs).pack(side=tk.LEFT)

        # Panel derecho: resumen y resultado
        right = ttk.Frame(paned, width=380)
        paned.add(right, weight=2)

        # Resultado grande arriba
        res_frame = ttk.Frame(right)
        res_frame.pack(fill=tk.X, padx=6, pady=(6, 4))
        ttk.Label(res_frame, text='Resultado', font=('Arial', 10, 'bold')).pack(anchor='w')
        self.result_label = ttk.Label(res_frame, text='(aún no calculado)', anchor='w', justify='left')
        # usar fuente grande y negrita para destacar
        self.result_label.config(font=('Arial', 16, 'bold'))
        self.result_label.pack(fill=tk.X, pady=(4, 6))

        # Resumen JSON con scrollbar
        summ_frame = ttk.LabelFrame(right, text="Resumen JSON")
        summ_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.summary_text = tk.Text(summ_frame, wrap='word', height=18)
        self.summary_text.configure(state='disabled')
        vsb = ttk.Scrollbar(summ_frame, orient='vertical', command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=vsb.set)
        self.summary_text.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        summ_frame.rowconfigure(0, weight=1)
        summ_frame.columnconfigure(0, weight=1)

        # estado interno
        self.entries = {}
        self.centros = None
        self.W = None
        self.errorMSE = None
        self.error_permitido = None
        self.codificaciones = {}
        self.columnas = []
        self.input_cols = []
        self.output_cols = []

    def _limpiar_inputs(self):
        # limpia los inputs que se han creado
        for w in list(self.form_frame.winfo_children()):
            w.destroy()
        self.entries.clear()
        self.result_label.config(text='(aún no calculado)')

    def _set_summary(self, obj):
        text = json.dumps(obj, indent=2, ensure_ascii=False)
        self.summary_text.configure(state='normal')
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state='disabled')

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
        # prioridad a input_names
        input_cols = data.get('input_names') or resumen.get('input_names')
        output_cols = data.get('output_names') or resumen.get('output_names') or data.get('output_name') or resumen.get('output_name')
        columns = resumen.get('columns') or data.get('columns')
        entradas = resumen.get('entradas') or data.get('entradas')

        if not input_cols:
            if isinstance(columns, list) and columns:
                if len(columns) > 1:
                    input_cols = columns[:-1]
                    if not output_cols:
                        output_cols = [columns[-1]]
                else:
                    input_cols = columns
            elif entradas:
                try:
                    n = int(entradas)
                    input_cols = [f'X{i+1}' for i in range(n)]
                except Exception:
                    input_cols = []
            else:
                input_cols = []

        if not output_cols and isinstance(columns, list) and columns:
            if set(input_cols) and all(c in columns for c in input_cols):
                output_cols = [c for c in columns if c not in input_cols]
            if not output_cols and len(columns) > len(input_cols):
                output_cols = [columns[-1]]

        centros_json = data.get('centros_radiales') or data.get('centros') or data.get('radios') or data.get('radios_centros')
        if centros_json:
            try:
                self.centros = np.array(centros_json, dtype=float)
            except Exception:
                self.centros = None

        pesos_json = data.get('pesos')
        self.W = _pesos_from_json(pesos_json)

        self.errorMSE = data.get('errorMSE') or resumen.get('errorMSE') or resumen.get('error_optimo')
        self.error_permitido = resumen.get('error_optimo') or data.get('error_optimo')
        self.codificaciones = data.get('codificaciones') or data.get('mapeos') or {}
        self.columnas = columns or []
        self.input_cols = input_cols or []
        self.output_cols = output_cols or []

        # mostrar resumen
        self._set_summary(resumen if resumen else data)

        # limpiar e insertar inputs con grid (alineado)
        for w in list(self.form_frame.winfo_children()):
            w.destroy()
        self.entries.clear()

        if not self.input_cols:
            ttk.Label(self.form_frame, text='No se encontraron nombres de entradas (input_names) en el JSON.').grid(row=0, column=0, sticky='w')
            return

        # crear labels+entries en grid con columnas fijas
        for i, col in enumerate(self.input_cols):
            lbl = ttk.Label(self.form_frame, text=f'{col}:')
            ent_var = tk.StringVar()
            ent = ttk.Entry(self.form_frame, textvariable=ent_var, width=18)
            lbl.grid(row=i, column=0, padx=(0, 8), pady=4, sticky='w')
            ent.grid(row=i, column=1, padx=(0, 4), pady=4, sticky='w')
            self.entries[col] = ent_var

        # info resumen justo debajo de inputs
        info = 'Entradas: ' + ', '.join(self.input_cols)
        if self.output_cols:
            info += '\nSalida(s): ' + ', '.join(self.output_cols)
        ttk.Label(self.form_frame, text=info).grid(row=len(self.input_cols), column=0, columnspan=2, pady=(8, 4), sticky='w')

    def calcular_salida(self):
        # validaciones simples
        if self.W is None or self.centros is None:
            messagebox.showwarning('Aviso', 'No hay pesos o centros cargados en el JSON.')
            return

        try:
            entradas_usuario = []
            for col in self.input_cols:
                s = self.entries[col].get()
                if s == '':
                    raise ValueError(f'La entrada "{col}" está vacía.')
                entradas_usuario.append(float(s))

            x = np.array(entradas_usuario, dtype=float).reshape(-1)  # vector (n_entradas,)

            if x.shape[0] != self.centros.shape[1]:
                raise ValueError(f'Número de entradas ({x.shape[0]}) no coincide con centros (esperan {self.centros.shape[1]}).')

            D = np.linalg.norm(self.centros - x, axis=1)
            D_safe = np.where(D <= 0, 1e-6, D)
            FA = (D_safe ** 2) * np.log(D_safe)

            phi = np.concatenate(([1.0], FA))

            W = self.W.flatten()
            if W.shape[0] != phi.shape[0]:
                if W.shape[0] > phi.shape[0]:
                    W = W[:phi.shape[0]]
                else:
                    raise ValueError(f'Longitud de pesos ({W.shape[0]}) no coincide con phi ({phi.shape[0]}).')

            salida_valor = float(np.dot(phi, W))

            salida_nombre = self.output_cols[0] if self.output_cols else 'Salida'
            codif = self.codificaciones.get(salida_nombre, {}) if isinstance(self.codificaciones, dict) else {}

            if isinstance(codif, dict) and codif:
                inverso = {float(v): k for k, v in codif.items()}
                codigo_closer = min(inverso.keys(), key=lambda c: abs(c - salida_valor))
                categoria = inverso[codigo_closer]
                mensaje = f'Salida estimada ({salida_nombre}): {salida_valor:.6f}   Categoría: {categoria}'
                # color según categoría si coincide con palabras comunes
                color = '#0b6b0b'  # verde por defecto
                if isinstance(categoria, str):
                    cat_low = categoria.lower()
                    if 'alta' in cat_low or 'alto' in cat_low:
                        color = '#b20000'
                    elif 'media' in cat_low or 'med' in cat_low:
                        color = '#b26b00'
            else:
                mensaje = f'Salida estimada ({salida_nombre}): {salida_valor:.6f}'
                color = '#0b6b0b'

            # Mostrar resultado grande y en negrita en el panel derecho
            self.result_label.config(text=mensaje, foreground=color)

        except Exception as e:
            messagebox.showerror('Error en simulación', str(e))
