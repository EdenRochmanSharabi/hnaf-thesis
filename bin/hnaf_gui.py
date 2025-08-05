#!/usr/bin/env python3
"""
Interfaz gráfica para el control de hiperparámetros de HNAF
Interfaz profesional para tesis de investigación
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
from hnaf_implementation import HNAF, train_hnaf, visualize_training
from src.hnaf_stable import StableHNAF, train_stable_hnaf
from src.hnaf_improved import ImprovedHNAF, train_improved_hnaf

class RedirectText:
    """Clase para redirigir la salida de print a un widget de texto"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update()

    def flush(self):
        pass

class HNAFGUI:
    def __init__(self, root):
        print("DEBUG: Inicializando HNAFGUI")
        self.root = root
        self.root.title("HNAF - Control de Hiperparámetros")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        print("DEBUG: Ventana configurada")
        
        # Variables para almacenar resultados
        self.training_results = None
        self.hnaf_model = None
        
        # Configurar estilo
        print("DEBUG: Configurando estilos")
        self.setup_styles()
        
        # Crear interfaz
        print("DEBUG: Creando widgets")
        self.create_widgets()
        
        # Configurar redirección de salida
        print("DEBUG: Configurando redirección de salida")
        self.redirect_output()
        print("DEBUG: HNAFGUI inicializada completamente")
    
    def setup_styles(self):
        """Configurar estilos profesionales"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores y estilos
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        
        # Configurar frames
        style.configure('Main.TFrame', background='#f0f0f0')
        style.configure('Panel.TFrame', background='white', relief='raised', borderwidth=1)
    
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="HNAF - Hybrid Normalized Advantage Function", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Frame superior para parámetros
        params_frame = ttk.Frame(main_frame, style='Panel.TFrame')
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Crear secciones de parámetros
        self.create_network_params(params_frame)
        self.create_training_params(params_frame)
        self.create_control_buttons(params_frame)
        
        # Frame medio para funciones personalizadas
        custom_frame = ttk.Frame(main_frame, style='Panel.TFrame')
        custom_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Crear sección de funciones personalizadas
        self.create_custom_functions_section(custom_frame)
        
        # Frame inferior para resultados
        results_frame = ttk.Frame(main_frame, style='Panel.TFrame')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear secciones de resultados
        self.create_output_section(results_frame)
        self.create_plots_section(results_frame)
    
    def create_network_params(self, parent):
        """Crear sección de parámetros de red"""
        # Frame para parámetros de red
        network_frame = ttk.LabelFrame(parent, text="Parámetros de Red Neuronal", padding=10)
        network_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Variables para parámetros de red (valores optimizados por defecto)
        self.state_dim_var = tk.IntVar(value=2)
        self.action_dim_var = tk.IntVar(value=2)
        self.num_modes_var = tk.IntVar(value=2)
        self.hidden_dim_var = tk.IntVar(value=64)  # Optimizado
        self.num_layers_var = tk.IntVar(value=3)   # Optimizado
        
        # Crear controles
        ttk.Label(network_frame, text="Dimensión del Estado:", style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=1, to=10, textvariable=self.state_dim_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Dimensión de Acción:", style='Info.TLabel').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=1, to=10, textvariable=self.action_dim_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Número de Modos:", style='Info.TLabel').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=1, to=5, textvariable=self.num_modes_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Capas Ocultas:", style='Info.TLabel').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=16, to=256, increment=16, textvariable=self.hidden_dim_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Número de Capas:", style='Info.TLabel').grid(row=4, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=2, to=5, textvariable=self.num_layers_var, width=10).grid(row=4, column=1, padx=5, pady=2)
    
    def create_training_params(self, parent):
        """Crear sección de parámetros de entrenamiento"""
        # Frame para parámetros de entrenamiento
        training_frame = ttk.LabelFrame(parent, text="Parámetros de Entrenamiento", padding=10)
        training_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Variables para parámetros de entrenamiento (valores optimizados por defecto)
        self.learning_rate_var = tk.DoubleVar(value=0.0001)  # 1e-4
        self.tau_var = tk.DoubleVar(value=0.001)            # 0.001
        self.gamma_var = tk.DoubleVar(value=0.9)            # 0.9
        self.num_episodes_var = tk.IntVar(value=1000)       # 1000
        self.batch_size_var = tk.IntVar(value=32)           # 32
        self.initial_epsilon_var = tk.DoubleVar(value=0.5)  # ε inicial optimizado
        self.final_epsilon_var = tk.DoubleVar(value=0.05)   # ε final optimizado
        self.max_steps_var = tk.IntVar(value=50)            # Horizonte optimizado
        
        # Crear controles
        ttk.Label(training_frame, text="Learning Rate:", style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=0.0001, to=0.1, increment=0.0001, textvariable=self.learning_rate_var, 
                   format="%.4f", width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Tau (Soft Update):", style='Info.TLabel').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=0.0001, to=0.01, increment=0.0001, textvariable=self.tau_var, 
                   format="%.4f", width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Gamma (Discount):", style='Info.TLabel').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=0.8, to=0.999, increment=0.001, textvariable=self.gamma_var, 
                   format="%.3f", width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Episodios:", style='Info.TLabel').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=100, to=5000, increment=100, textvariable=self.num_episodes_var, 
                   width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Batch Size:", style='Info.TLabel').grid(row=4, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=8, to=128, increment=8, textvariable=self.batch_size_var, 
                   width=10).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="ε Inicial:", style='Info.TLabel').grid(row=5, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.initial_epsilon_var, 
                   format="%.2f", width=10).grid(row=5, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="ε Final:", style='Info.TLabel').grid(row=6, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=0.01, to=0.2, increment=0.01, textvariable=self.final_epsilon_var, 
                   format="%.2f", width=10).grid(row=6, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Max Steps:", style='Info.TLabel').grid(row=7, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=20, to=100, increment=10, textvariable=self.max_steps_var, 
                   width=10).grid(row=7, column=1, padx=5, pady=2)
    
    def create_custom_functions_section(self, parent):
        """Crear sección para funciones personalizadas"""
        # Frame principal para funciones personalizadas
        custom_main_frame = ttk.LabelFrame(parent, text="Funciones Personalizadas", padding=10)
        custom_main_frame.pack(fill=tk.X)
        
        # Frame izquierdo para editor de código
        editor_frame = ttk.Frame(custom_main_frame)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Label para el editor
        ttk.Label(editor_frame, text="Editor de Funciones:", style='Header.TLabel').pack(anchor='w')
        
        # Editor de código
        print("DEBUG: Creando editor de código")
        self.code_editor = scrolledtext.ScrolledText(editor_frame, height=15, width=80, 
                                                   font=('Consolas', 9), bg='#f8f8f8', fg='black')
        self.code_editor.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        print("DEBUG: Editor de código creado y empaquetado")
        
        # Insertar plantilla por defecto
        print("DEBUG: Llamando a insert_default_template")
        self.insert_default_template()
        
        # Frame derecho para controles
        controls_frame = ttk.Frame(custom_main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de control
        ttk.Button(controls_frame, text="Cargar Plantilla", 
                  command=self.insert_default_template).pack(pady=5, fill=tk.X)
        
        ttk.Button(controls_frame, text="Probar Funciones", 
                  command=self.test_custom_functions).pack(pady=5, fill=tk.X)
        
        ttk.Button(controls_frame, text="Guardar Funciones", 
                  command=self.save_custom_functions).pack(pady=5, fill=tk.X)
        
        ttk.Button(controls_frame, text="Cargar Funciones", 
                  command=self.load_custom_functions).pack(pady=5, fill=tk.X)
        
        # Checkbox para usar funciones personalizadas
        self.use_custom_functions_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Usar Funciones Personalizadas", 
                       variable=self.use_custom_functions_var).pack(pady=10)
        
        # Label de estado
        self.custom_status_label = ttk.Label(controls_frame, text="Listo", style='Info.TLabel')
        self.custom_status_label.pack(pady=5)
    
    def insert_default_template(self):
        """Insertar plantilla por defecto en el editor"""
        print("DEBUG: Botón 'Cargar Plantilla' presionado")
        template = '''import numpy as np

# Coordenadas iniciales
x0, y0 = 1, 1  # puedes cambiar estos valores

# Definimos los vectores x1 y x2
A1 = np.array([[1, 50],
               [-1, 1]])
x1 = A1 @ np.array([[x0], [y0]])

A2 = np.array([[1, -1],
               [50, 1]])
x2 = A2 @ np.array([[x0], [y0]])

# Imprimir resultados de las transformaciones
print("x1 =\\n", x1)
print("x2 =\\n", x2)

# Función recompensa: minimizar diferencia de distancias al origen
# Reward = | ||(x, y)|| - ||(x0, y0)|| |
def reward(x, y, x0, y0):
    norm_xy = np.linalg.norm([x, y])
    norm_x0y0 = np.linalg.norm([x0, y0])
    return abs(norm_xy - norm_x0y0)

# Ejemplo de uso
x, y = 3, 4  # ejemplo de punto
r = reward(x, y, x0, y0)
print("Recompensa:", r)

# Funciones de transformación para cada modo
def transform_mode_0(x0, y0):
    """Transformación para el modo 0"""
    A = np.array([[1, 50],
                  [-1, 1]])
    result = A @ np.array([[x0], [y0]])
    return result[0, 0], result[1, 0]

def transform_mode_1(x0, y0):
    """Transformación para el modo 1"""
    A = np.array([[1, -1],
                  [50, 1]])
    result = A @ np.array([[x0], [y0]])
    return result[0, 0], result[1, 0]

# Lista de funciones de transformación (una por modo)
transformation_functions = [transform_mode_0, transform_mode_1]

# Función de recompensa
def reward_function(x, y, x0, y0):
    """Función de recompensa personalizada"""
    norm_xy = np.linalg.norm([x, y])
    norm_x0y0 = np.linalg.norm([x0, y0])
    return abs(norm_xy - norm_x0y0)
'''
        
        print("DEBUG: Limpiando editor")
        self.code_editor.delete(1.0, tk.END)
        print("DEBUG: Insertando plantilla")
        self.code_editor.insert(1.0, template)
        print("DEBUG: Forzando actualización del editor")
        self.code_editor.update()
        self.code_editor.see(1.0)  # Mover cursor al inicio
        print("DEBUG: Plantilla cargada exitosamente")
        print("DEBUG: Contenido del editor después de cargar:")
        content = self.code_editor.get(1.0, tk.END)
        print("DEBUG: Longitud del contenido:", len(content))
        print("DEBUG: Primeras 100 caracteres:", content[:100])
    
    def test_custom_functions(self):
        """Probar las funciones personalizadas"""
        print("DEBUG: Botón 'Probar Funciones' presionado")
        try:
            # Obtener código del editor
            code = self.code_editor.get(1.0, tk.END)
            print("DEBUG: Código obtenido del editor")
            
            # Crear namespace local
            local_namespace = {}
            
            # Ejecutar código
            exec(code, globals(), local_namespace)
            print("DEBUG: Código ejecutado exitosamente")
            
            # Verificar que las funciones necesarias existen
            required_functions = ['transformation_functions', 'reward_function']
            missing_functions = [func for func in required_functions if func not in local_namespace]
            
            if missing_functions:
                raise ValueError(f"Faltan las siguientes funciones: {missing_functions}")
            
            # Probar las funciones
            print("="*60)
            print("PRUEBA DE FUNCIONES PERSONALIZADAS")
            print("="*60)
            
            # Probar transformaciones
            x0, y0 = 1, 1
            transformation_functions = local_namespace['transformation_functions']
            
            print(f"Estado inicial: ({x0}, {y0})")
            for i, transform_func in enumerate(transformation_functions):
                try:
                    x1, y1 = transform_func(x0, y0)
                    print(f"Modo {i}: ({x1:.4f}, {y1:.4f})")
                except Exception as e:
                    print(f"Error en transformación del modo {i}: {e}")
            
            # Probar función de recompensa
            reward_func = local_namespace['reward_function']
            test_x, test_y = 3, 4
            try:
                reward = reward_func(test_x, test_y, x0, y0)
                print(f"Recompensa para ({test_x}, {test_y}): {reward:.4f}")
            except Exception as e:
                print(f"Error en función de recompensa: {e}")
            
            print("="*60)
            print("PRUEBA COMPLETADA EXITOSAMENTE")
            print("="*60)
            
            self.custom_status_label.config(text="Funciones válidas")
            
        except Exception as e:
            error_msg = f"Error al probar funciones: {str(e)}"
            print(error_msg)
            self.custom_status_label.config(text="Error en funciones")
            messagebox.showerror("Error", error_msg)
    
    def save_custom_functions(self):
        """Guardar funciones personalizadas en archivo"""
        print("DEBUG: Botón 'Guardar Funciones' presionado")
        try:
            code = self.code_editor.get(1.0, tk.END)
            print("DEBUG: Código obtenido del editor")
            
            # Guardar en archivo
            with open('custom_functions.py', 'w') as f:
                f.write(code)
            
            print("DEBUG: Archivo guardado exitosamente")
            print("Funciones guardadas en 'custom_functions.py'")
            self.custom_status_label.config(text="Funciones guardadas")
            
        except Exception as e:
            error_msg = f"Error al guardar: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def load_custom_functions(self):
        """Cargar funciones personalizadas desde archivo"""
        print("DEBUG: Botón 'Cargar Funciones' presionado")
        try:
            with open('custom_functions.py', 'r') as f:
                code = f.read()
            print("DEBUG: Archivo leído exitosamente")
            
            self.code_editor.delete(1.0, tk.END)
            self.code_editor.insert(1.0, code)
            print("DEBUG: Código cargado en el editor")
            
            print("Funciones cargadas desde 'custom_functions.py'")
            self.custom_status_label.config(text="Funciones cargadas")
            
        except FileNotFoundError:
            print("DEBUG: Archivo no encontrado")
            print("Archivo 'custom_functions.py' no encontrado")
            self.custom_status_label.config(text="Archivo no encontrado")
        except Exception as e:
            error_msg = f"Error al cargar: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def create_control_buttons(self, parent):
        """Crear botones de control"""
        # Frame para botones
        button_frame = ttk.Frame(parent)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        
        # Botón de entrenamiento
        self.train_button = ttk.Button(button_frame, text="Iniciar Entrenamiento", 
                                      command=self.start_training, style='Accent.TButton')
        self.train_button.pack(pady=5, fill=tk.X)
        
        # Botón de evaluación
        self.eval_button = ttk.Button(button_frame, text="Evaluar Modelo", 
                                     command=self.evaluate_model, state=tk.DISABLED)
        self.eval_button.pack(pady=5, fill=tk.X)
        
        # Botón de verificación
        self.verify_button = ttk.Button(button_frame, text="Verificar HNAF", 
                                       command=self.verify_hnaf, state=tk.DISABLED)
        self.verify_button.pack(pady=5, fill=tk.X)
        
        # Botón de limpiar
        self.clear_button = ttk.Button(button_frame, text="Limpiar Salida", 
                                      command=self.clear_output)
        self.clear_button.pack(pady=5, fill=tk.X)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(pady=5, fill=tk.X)
        
        # Label de estado
        self.status_label = ttk.Label(button_frame, text="Listo", style='Info.TLabel')
        self.status_label.pack(pady=5)
    
    def create_output_section(self, parent):
        """Crear sección de salida de texto"""
        # Frame para salida
        output_frame = ttk.LabelFrame(parent, text="Salida de Terminal", padding=10)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Área de texto con scroll
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=60, 
                                                   font=('Consolas', 9), bg='black', fg='white')
        self.output_text.pack(fill=tk.BOTH, expand=True)
    
    def create_plots_section(self, parent):
        """Crear sección de gráficos"""
        # Frame para gráficos
        plots_frame = ttk.LabelFrame(parent, text="Gráficos de Resultados", padding=10)
        plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurar gráfico inicial
        self.ax.set_title("Resultados del Entrenamiento")
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def redirect_output(self):
        """Redirigir la salida de print al widget de texto"""
        self.redirect = RedirectText(self.output_text)
        sys.stdout = self.redirect
    
    def start_training(self):
        """Iniciar entrenamiento en un hilo separado"""
        print("DEBUG: Botón 'Iniciar Entrenamiento' presionado")
        if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.is_alive():
            print("DEBUG: Entrenamiento ya en progreso")
            messagebox.showwarning("Advertencia", "El entrenamiento ya está en progreso")
            return
        
        # Obtener parámetros
        params = self.get_training_parameters()
        print("DEBUG: Parámetros obtenidos:", params)
        
        # Verificar si usar funciones personalizadas
        if self.use_custom_functions_var.get():
            try:
                # Cargar funciones personalizadas
                code = self.code_editor.get(1.0, tk.END)
                local_namespace = {}
                exec(code, globals(), local_namespace)
                
                # Verificar funciones requeridas
                if 'transformation_functions' not in local_namespace or 'reward_function' not in local_namespace:
                    raise ValueError("Faltan las funciones 'transformation_functions' o 'reward_function'")
                
                params['custom_functions'] = local_namespace
                print("Usando funciones personalizadas")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error en funciones personalizadas: {str(e)}")
                return
        
        # Actualizar interfaz
        self.train_button.config(state=tk.DISABLED)
        self.status_label.config(text="Entrenando...")
        self.progress_var.set(0)
        
        # Iniciar entrenamiento en hilo separado
        self.training_thread = threading.Thread(target=self.run_training, args=(params,))
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def get_training_parameters(self):
        """Obtener parámetros de la interfaz"""
        return {
            'state_dim': self.state_dim_var.get(),
            'action_dim': self.action_dim_var.get(),
            'num_modes': self.num_modes_var.get(),
            'hidden_dim': self.hidden_dim_var.get(),
            'num_layers': self.num_layers_var.get(),
            'lr': self.learning_rate_var.get(),
            'tau': self.tau_var.get(),
            'gamma': self.gamma_var.get(),
            'num_episodes': self.num_episodes_var.get(),
            'batch_size': self.batch_size_var.get(),
            'initial_epsilon': self.initial_epsilon_var.get(),
            'final_epsilon': self.final_epsilon_var.get(),
            'max_steps': self.max_steps_var.get()
        }
    
    def run_training(self, params):
        """Ejecutar entrenamiento con parámetros dados"""
        try:
            print("="*60)
            print("INICIANDO ENTRENAMIENTO HNAF MEJORADO")
            print("="*60)
            print("🚀 Usando HNAF MEJORADO con optimizaciones avanzadas:")
            print(f"  - Red: {params['num_layers']} capas de {params['hidden_dim']} unidades")
            print(f"  - ε-greedy decay: {params['initial_epsilon']} -> {params['final_epsilon']}")
            print(f"  - Learning rate: {params['lr']}")
            print(f"  - Prioritized replay con buffer de 10000")
            print(f"  - Normalización de estados y recompensas")
            print(f"  - Reward shaping local")
            print(f"  - Evaluación en grid 100x100")
            print(f"  - Horizonte más largo: {params['max_steps']} pasos")
            
            if 'custom_functions' in params:
                print("✅ Usando funciones personalizadas")
            else:
                print("✅ Usando funciones por defecto")
            print()
            
            # Fijar semilla para reproducibilidad
            np.random.seed(42)
            torch.manual_seed(42)
            
            # Crear modelo HNAF mejorado
            self.hnaf_model = ImprovedHNAF(
                state_dim=params['state_dim'],
                action_dim=params['action_dim'],
                num_modes=params['num_modes'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                lr=params['lr'],
                tau=params['tau'],
                gamma=params['gamma']
            )
            
            # Si hay funciones personalizadas, actualizar el modelo
            if 'custom_functions' in params:
                custom_funcs = params['custom_functions']
                self.hnaf_model.transformation_functions = custom_funcs['transformation_functions']
                self.hnaf_model.reward_function = custom_funcs['reward_function']
            
            # Métricas de entrenamiento
            episode_rewards = []
            losses = []
            eval_rewards = []
            grid_accuracies = []
            eval_interval = 50
            
            # Entrenamiento mejorado con ε-greedy decay
            epsilon_decay = (params['initial_epsilon'] - params['final_epsilon']) / params['num_episodes']
            
            for episode in range(params['num_episodes']):
                # Calcular epsilon actual
                epsilon = max(params['final_epsilon'], 
                            params['initial_epsilon'] - episode * epsilon_decay)
                
                # Entrenar episodio
                reward, _ = self.hnaf_model.train_episode(
                    max_steps=params['max_steps'],
                    epsilon=epsilon
                )
                episode_rewards.append(reward)
                
                # Actualizar redes
                loss = self.hnaf_model.update(batch_size=params['batch_size'])
                if loss is not None:
                    losses.append(loss)
                
                # Actualizar redes objetivo
                if hasattr(self.hnaf_model, 'update_target_networks'):
                    self.hnaf_model.update_target_networks()
                
                # Actualizar progreso
                progress = (episode + 1) / params['num_episodes'] * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                # Evaluación periódica
                if (episode + 1) % eval_interval == 0:
                    eval_reward, mode_selections = self.hnaf_model.evaluate_policy(num_episodes=10)
                    eval_rewards.append(eval_reward)
                    
                    # Evaluación en grid para HNAF mejorado
                    if hasattr(self.hnaf_model, 'evaluate_policy_grid'):
                        grid_results = self.hnaf_model.evaluate_policy_grid(grid_size=50)
                        grid_accuracies.append(grid_results['optimal_accuracy'])
                        
                        print(f"Episodio {episode+1}/{params['num_episodes']}")
                        print(f"  ε: {epsilon:.3f}")
                        print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
                        print(f"  Recompensa evaluación: {eval_reward:.4f}")
                        print(f"  Precisión grid: {grid_results['optimal_accuracy']:.2%}")
                        print(f"  Selección modos: {mode_selections}")
                        if losses:
                            print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
                        print()
                    else:
                        print(f"Episodio {episode+1}/{params['num_episodes']}")
                        print(f"  Recompensa promedio: {np.mean(episode_rewards[-eval_interval:]):.4f}")
                        print(f"  Recompensa evaluación: {eval_reward:.4f}")
                        print(f"  Selección de modos: {mode_selections}")
                        if losses:
                            print(f"  Pérdida promedio: {np.mean(losses[-eval_interval:]):.6f}")
                        print()
            
            # Verificación final
            self.hnaf_model.verify_hnaf()
            
            # Guardar resultados
            self.training_results = {
                'episode_rewards': episode_rewards,
                'losses': losses,
                'eval_rewards': eval_rewards,
                'grid_accuracies': grid_accuracies,
                'eval_interval': eval_interval
            }
            
            # Actualizar interfaz
            self.root.after(0, self.training_completed)
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")
            self.root.after(0, self.training_error, str(e))
    
    def training_completed(self):
        """Llamado cuando el entrenamiento se completa"""
        self.train_button.config(state=tk.NORMAL)
        self.eval_button.config(state=tk.NORMAL)
        self.verify_button.config(state=tk.NORMAL)
        self.status_label.config(text="Entrenamiento completado")
        self.progress_var.set(100)
        
        # Actualizar gráficos
        self.update_plots()
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
    
    def training_error(self, error_msg):
        """Llamado cuando hay un error en el entrenamiento"""
        self.train_button.config(state=tk.NORMAL)
        self.status_label.config(text="Error en entrenamiento")
        messagebox.showerror("Error", f"Error durante el entrenamiento:\n{error_msg}")
    
    def evaluate_model(self):
        """Evaluar el modelo entrenado"""
        print("DEBUG: Botón 'Evaluar Modelo' presionado")
        if self.hnaf_model is None:
            print("DEBUG: No hay modelo entrenado")
            messagebox.showwarning("Advertencia", "No hay modelo entrenado para evaluar")
            return
        
        print("DEBUG: Iniciando evaluación del modelo")
        print("\n" + "="*60)
        print("EVALUACIÓN DEL MODELO")
        print("="*60)
        
        # Evaluación estándar
        eval_reward, mode_selections = self.hnaf_model.evaluate_policy(num_episodes=20)
        
        print(f"Recompensa promedio de evaluación: {eval_reward:.4f}")
        print(f"Distribución de selección de modos: {mode_selections}")
        
        # Evaluación en grid si está disponible
        if hasattr(self.hnaf_model, 'evaluate_policy_grid'):
            print("\n📊 Evaluación en grid 100x100:")
            grid_results = self.hnaf_model.evaluate_policy_grid(grid_size=100)
            print(f"Precisión en grid: {grid_results['optimal_accuracy']:.2%}")
            print(f"Q-values promedio: {np.mean(grid_results['q_values']):.4f}")
            print(f"Q-values std: {np.std(grid_results['q_values']):.4f}")
        
        print("="*60)
    
    def verify_hnaf(self):
        """Verificar el funcionamiento del HNAF"""
        print("DEBUG: Botón 'Verificar HNAF' presionado")
        if self.hnaf_model is None:
            print("DEBUG: No hay modelo entrenado")
            messagebox.showwarning("Advertencia", "No hay modelo entrenado para verificar")
            return
        
        print("DEBUG: Iniciando verificación HNAF")
        print("\n" + "="*60)
        print("VERIFICACIÓN HNAF")
        print("="*60)
        
        self.hnaf_model.verify_hnaf()
    
    def update_plots(self):
        """Actualizar gráficos con los resultados del entrenamiento"""
        if self.training_results is None:
            return
        
        # Limpiar gráfico
        self.ax.clear()
        
        # Datos
        episode_rewards = self.training_results['episode_rewards']
        eval_rewards = self.training_results['eval_rewards']
        eval_interval = self.training_results['eval_interval']
        grid_accuracies = self.training_results.get('grid_accuracies', [])
        
        # Crear subplots si hay métricas adicionales
        if grid_accuracies:
            # Crear figura con subplots
            self.fig.clear()
            self.ax1 = self.fig.add_subplot(2, 1, 1)
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            
            # Gráfico superior: recompensas
            self.ax1.plot(episode_rewards, alpha=0.6, label='Episodio', color='blue')
            
            if eval_rewards:
                eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
                self.ax1.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluación')
            
            # Promedio móvil
            window = min(100, len(episode_rewards) // 10)
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                self.ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'g-', 
                            linewidth=2, label=f'Promedio móvil ({window})')
            
            self.ax1.set_title("Recompensas del Entrenamiento")
            self.ax1.set_ylabel("Recompensa")
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            # Gráfico inferior: precisión en grid
            eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
            self.ax2.plot(eval_episodes, grid_accuracies, 'purple', linewidth=2, marker='o')
            self.ax2.set_title("Precisión en Grid de Evaluación")
            self.ax2.set_xlabel("Episodio")
            self.ax2.set_ylabel("Precisión")
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(0, 1)
            
        else:
            # Gráfico simple para HNAF estable
            self.ax.plot(episode_rewards, alpha=0.6, label='Episodio', color='blue')
            
            if eval_rewards:
                eval_episodes = np.arange(eval_interval, len(episode_rewards) + 1, eval_interval)
                self.ax.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluación')
            
            # Promedio móvil
            window = min(100, len(episode_rewards) // 10)
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                self.ax.plot(range(window-1, len(episode_rewards)), moving_avg, 'g-', 
                            linewidth=2, label=f'Promedio móvil ({window})')
            
            self.ax.set_title("Resultados del Entrenamiento HNAF")
            self.ax.set_xlabel("Episodio")
            self.ax.set_ylabel("Recompensa")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
        
        # Actualizar canvas
        self.canvas.draw()
    
    def clear_output(self):
        """Limpiar la salida de texto"""
        print("DEBUG: Botón 'Limpiar Salida' presionado")
        self.output_text.delete(1.0, tk.END)
        print("DEBUG: Salida limpiada")

def main():
    """Función principal"""
    print("DEBUG: Iniciando aplicación HNAF GUI")
    root = tk.Tk()
    print("DEBUG: Ventana principal creada")
    app = HNAFGUI(root)
    print("DEBUG: Interfaz HNAF creada")
    
    # Configurar cierre limpio
    def on_closing():
        print("DEBUG: Cerrando aplicación")
        sys.stdout = sys.__stdout__  # Restaurar stdout
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    print("DEBUG: Iniciando mainloop")
    root.mainloop()
    print("DEBUG: Aplicación cerrada")

if __name__ == "__main__":
    main() 