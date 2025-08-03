#!/usr/bin/env python3
"""
Interfaz gráfica principal para HNAF
Maneja la GUI y coordina con los managers de entrenamiento y visualización
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import io

# Imports de visualización (solo cuando se necesiten)
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib no disponible. Gráficos deshabilitados.")

import numpy as np


from final_app.training_manager import TrainingManager
from final_app.config_manager import get_config_manager
from final_app.logging_manager import get_logger, log_exception, log_info, log_warning

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
    def __init__(self, root, config_manager, cli_mode=False):
        """Inicializa la GUI SIN valores hardcodeados."""
        self.root = root
        self.config_manager = config_manager
        self.cli_mode = cli_mode
        if not cli_mode:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.training_results = None
        self.training_thread = None

        self.hnaf_model = None
        self.logger = get_logger("HNAFGUI")

        # Cargar configuraciones específicas (necesario para ambos modos)
        self.interface_config = self.config_manager.get_interface_config()
        self.checkbox_defaults = self.interface_config['checkboxes']
        self.defaults_config = self.config_manager.get_defaults_config()
        
        # Inicializar variables de checkboxes desde configuración
        self.show_rewards_var = tk.BooleanVar(value=self.checkbox_defaults['show_rewards'])
        self.show_precision_var = tk.BooleanVar(value=self.checkbox_defaults['show_precision'])
        self.show_loss_var = tk.BooleanVar(value=self.checkbox_defaults['show_loss'])

        # Solo crear GUI si no estamos en modo CLI
        if not self.cli_mode:
            self.setup_styles()
            self.create_widgets()
            
            # Configurar redirección de salida
            print("DEBUG: Configurando redirección de salida")
            self.redirect_output()
            print("DEBUG: HNAFGUI inicializada completamente")
        else:
            # En modo CLI, solo inicializar variables necesarias
            self._init_cli_variables()
            print("DEBUG: HNAFGUI inicializada en modo CLI")
    
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
        self.create_plot_controls(params_frame)  # Añadir controles de gráficos
        
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
        
        # Variables para parámetros de red desde configuración
        network_defaults = self.config_manager.get_network_defaults()
        
        self.state_dim_var = tk.IntVar(value=network_defaults['state_dim'])
        self.action_dim_var = tk.IntVar(value=network_defaults['action_dim'])
        self.num_modes_var = tk.IntVar(value=network_defaults['num_modes'])
        self.hidden_dim_var = tk.IntVar(value=network_defaults['hidden_dim'])
        self.num_layers_var = tk.IntVar(value=network_defaults['num_layers'])
        
        # Crear controles con rangos desde configuración
        network_ranges = self.config_manager.get_network_ranges()
        
        ttk.Label(network_frame, text="Dimensión del Estado:", style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=network_ranges['state_dim'][0], to=network_ranges['state_dim'][1], 
                   textvariable=self.state_dim_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Dimensión de Acción:", style='Info.TLabel').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=network_ranges['action_dim'][0], to=network_ranges['action_dim'][1], 
                   textvariable=self.action_dim_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Número de Modos:", style='Info.TLabel').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=network_ranges['num_modes'][0], to=network_ranges['num_modes'][1], 
                   textvariable=self.num_modes_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Capas Ocultas:", style='Info.TLabel').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=network_ranges['hidden_dim'][0], to=network_ranges['hidden_dim'][1], 
                   increment=network_ranges['hidden_dim_increment'], textvariable=self.hidden_dim_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(network_frame, text="Número de Capas:", style='Info.TLabel').grid(row=4, column=0, sticky='w', pady=2)
        ttk.Spinbox(network_frame, from_=network_ranges['num_layers'][0], to=network_ranges['num_layers'][1], 
                   textvariable=self.num_layers_var, width=10).grid(row=4, column=1, padx=5, pady=2)
    
    def create_training_params(self, parent):
        """Crear sección de parámetros de entrenamiento"""
        # Frame para parámetros de entrenamiento
        training_frame = ttk.LabelFrame(parent, text="Parámetros de Entrenamiento", padding=10)
        training_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Variables para parámetros de entrenamiento desde configuración
        training_defaults = self.config_manager.get_training_defaults()
        
        self.learning_rate_var = tk.DoubleVar(value=training_defaults['learning_rate'])
        self.tau_var = tk.DoubleVar(value=training_defaults['tau'])
        self.gamma_var = tk.DoubleVar(value=training_defaults['gamma'])
        self.num_episodes_var = tk.IntVar(value=training_defaults['num_episodes'])
        self.batch_size_var = tk.IntVar(value=training_defaults['batch_size'])
        self.initial_epsilon_var = tk.DoubleVar(value=training_defaults['initial_epsilon'])
        self.final_epsilon_var = tk.DoubleVar(value=training_defaults['final_epsilon'])
        self.max_steps_var = tk.IntVar(value=training_defaults['max_steps'])
        
        # Crear controles con rangos desde configuración
        training_ranges = self.config_manager.get_training_ranges()
        
        ttk.Label(training_frame, text="Learning Rate:", style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['learning_rate'][0], to=training_ranges['learning_rate'][1], 
                   increment=training_ranges['learning_rate_increment'], textvariable=self.learning_rate_var, 
                   format="%.4f", width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Tau (Soft Update):", style='Info.TLabel').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['tau'][0], to=training_ranges['tau'][1], 
                   increment=training_ranges['tau_increment'], textvariable=self.tau_var, 
                   format="%.4f", width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Gamma (Discount):", style='Info.TLabel').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['gamma'][0], to=training_ranges['gamma'][1], 
                   increment=training_ranges['gamma_increment'], textvariable=self.gamma_var, 
                   format="%.3f", width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Episodios:", style='Info.TLabel').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['num_episodes'][0], to=training_ranges['num_episodes'][1], 
                   increment=training_ranges['episodes_increment'], textvariable=self.num_episodes_var, 
                   width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="Batch Size:", style='Info.TLabel').grid(row=4, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['batch_size'][0], to=training_ranges['batch_size'][1], 
                   increment=training_ranges['batch_size_increment'], textvariable=self.batch_size_var, 
                   width=10).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="ε Inicial:", style='Info.TLabel').grid(row=5, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['initial_epsilon'][0], to=training_ranges['initial_epsilon'][1], 
                   increment=training_ranges['epsilon_increment'], textvariable=self.initial_epsilon_var, 
                   format="%.2f", width=10).grid(row=5, column=1, padx=5, pady=2)
        
        ttk.Label(training_frame, text="ε Final:", style='Info.TLabel').grid(row=6, column=0, sticky='w', pady=2)
        ttk.Spinbox(training_frame, from_=training_ranges['final_epsilon'][0], to=training_ranges['final_epsilon'][1], 
                   increment=training_ranges['final_epsilon_increment'], textvariable=self.final_epsilon_var, 
                   format="%.2f", width=10).grid(row=6, column=1, padx=5, pady=2)
        
        # Max Steps
        max_steps_label = ttk.Label(training_frame, text="Max Steps:")
        max_steps_label.grid(row=7, column=0, sticky='w', padx=(10, 5), pady=2)
        
        # Usar variable ya definida arriba (convertir a StringVar si es necesario)
        self.max_steps_var = tk.StringVar(value=str(training_defaults['max_steps']))
        max_steps_spinbox = ttk.Spinbox(training_frame, from_=training_ranges['max_steps'][0], 
                                       to=training_ranges['max_steps'][1], textvariable=self.max_steps_var, width=10)
        max_steps_spinbox.grid(row=7, column=1, sticky='ew', padx=5, pady=2)

        # Buffer Capacity (Experience Replay)
        buffer_capacity_label = ttk.Label(training_frame, text="Buffer Capacity:")
        buffer_capacity_label.grid(row=8, column=0, sticky='w', padx=(10, 5), pady=2)
        
        self.buffer_capacity_var = tk.StringVar(value=str(training_defaults['buffer_capacity']))
        buffer_capacity_spinbox = ttk.Spinbox(training_frame, from_=training_ranges['buffer_capacity'][0], 
                                             to=training_ranges['buffer_capacity'][1], 
                                             increment=training_ranges['buffer_increment'], 
                                             textvariable=self.buffer_capacity_var, width=10)
        buffer_capacity_spinbox.grid(row=8, column=1, sticky='ew', padx=5, pady=2)

        # Alpha (Priority Exponent)
        alpha_label = ttk.Label(training_frame, text="Alpha (Priority):")
        alpha_label.grid(row=9, column=0, sticky='w', padx=(10, 5), pady=2)
        
        self.alpha_var = tk.StringVar(value=str(training_defaults['alpha']))
        alpha_spinbox = ttk.Spinbox(training_frame, from_=training_ranges['alpha'][0], 
                                   to=training_ranges['alpha'][1], 
                                   increment=training_ranges['priority_increment'], 
                                   textvariable=self.alpha_var, width=10)
        alpha_spinbox.grid(row=9, column=1, sticky='ew', padx=5, pady=2)

        # Beta (Bias Correction)
        beta_label = ttk.Label(training_frame, text="Beta (Bias Corr.):")
        beta_label.grid(row=10, column=0, sticky='w', padx=(10, 5), pady=2)
        
        self.beta_var = tk.StringVar(value=str(training_defaults['beta']))
        beta_spinbox = ttk.Spinbox(training_frame, from_=training_ranges['beta'][0], 
                                  to=training_ranges['beta'][1], 
                                  increment=training_ranges['priority_increment'], 
                                  textvariable=self.beta_var, width=10)
        beta_spinbox.grid(row=10, column=1, sticky='ew', padx=5, pady=2)

        # Supervised Episodes
        supervised_label = ttk.Label(training_frame, text="🧠 Supervised Episodes:")
        supervised_label.grid(row=11, column=0, sticky='w', padx=(10, 5), pady=2)
        
        self.supervised_episodes_var = tk.IntVar(value=training_defaults['supervised_episodes'])
        supervised_spinbox = ttk.Spinbox(training_frame, from_=training_ranges['supervised_episodes'][0], 
                                        to=training_ranges['supervised_episodes'][1], 
                                        increment=training_ranges['supervised_increment'], 
                                        textvariable=self.supervised_episodes_var, width=10)
        supervised_spinbox.grid(row=11, column=1, sticky='ew', padx=5, pady=2)
        
        # Etiqueta explicativa para entrenamiento supervisado
        supervised_info = ttk.Label(training_frame, text="(Entrenamiento balanceado: enseña ambos modos)", 
                                   font=('Arial', 8), foreground='gray')
        supervised_info.grid(row=11, column=2, sticky='w', padx=5, pady=2)

        # Reward Variance Control (desde configuración)
        reward_variance_label = ttk.Label(training_frame, text="Reward Normalize:")
        reward_variance_label.grid(row=12, column=0, sticky='w', padx=(10, 5), pady=2)
        
        self.reward_normalize_var = tk.BooleanVar(value=self.checkbox_defaults['reward_normalize'])
        reward_normalize_check = ttk.Checkbutton(training_frame, variable=self.reward_normalize_var)
        reward_normalize_check.grid(row=12, column=1, sticky='w', padx=5, pady=2)
        
        # Reward Shaping Control (desde configuración)
        reward_shaping_label = ttk.Label(training_frame, text="Reward Shaping:")
        reward_shaping_label.grid(row=13, column=0, sticky='w', padx=(10, 5), pady=2)
        
        self.reward_shaping_var = tk.BooleanVar(value=self.checkbox_defaults['reward_shaping'])
        reward_shaping_check = ttk.Checkbutton(training_frame, variable=self.reward_shaping_var)
        reward_shaping_check.grid(row=13, column=1, sticky='w', padx=5, pady=2)
        
        # Separador
        ttk.Separator(training_frame, orient='horizontal').grid(row=14, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Sección de optimización automática con Gemini
        optimization_frame = ttk.LabelFrame(training_frame, text="Optimización Automática con Gemini")
        optimization_frame.grid(row=15, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Checkbox para activar optimización automática
        self.use_gemini_optimization_var = tk.BooleanVar(value=self.checkbox_defaults['use_gemini_optimization'])
        gemini_check = ttk.Checkbutton(optimization_frame, text="Usar optimización automática con Gemini", 
                                      variable=self.use_gemini_optimization_var)
        gemini_check.pack(anchor='w', padx=10, pady=5)
        
        # Botones de control de optimización
        optimization_buttons_frame = ttk.Frame(optimization_frame)
        optimization_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_optimization_button = ttk.Button(optimization_buttons_frame, text="Iniciar Optimización", 
                                                   command=self.start_gemini_optimization)
        self.start_optimization_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_optimization_button = ttk.Button(optimization_buttons_frame, text="Detener Optimización", 
                                                  command=self.stop_gemini_optimization, state=tk.DISABLED)
        self.stop_optimization_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_best_params_button = ttk.Button(optimization_buttons_frame, text="Cargar Mejores Parámetros", 
                                                 command=self.load_best_params)
        self.load_best_params_button.pack(side=tk.LEFT)
        
        # Estado de optimización (desde configuración)
        status_messages = self.interface_config['status_messages']
        self.optimization_status_var = tk.StringVar(value=status_messages['optimization_inactive'])
        optimization_status_label = ttk.Label(optimization_frame, textvariable=self.optimization_status_var, 
                                            style='Info.TLabel')
        optimization_status_label.pack(anchor='w', padx=10, pady=2)
        
        # Progreso de optimización
        self.optimization_progress_var = tk.DoubleVar()
        self.optimization_progress_bar = ttk.Progressbar(optimization_frame, variable=self.optimization_progress_var, 
                                                       maximum=100)
        self.optimization_progress_bar.pack(fill=tk.X, padx=10, pady=2)
        
        # Separador
        ttk.Separator(training_frame, orient='horizontal').grid(row=15, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Sección de optimización automática con Optuna
        optuna_frame = ttk.LabelFrame(training_frame, text="Optimización Automática con Optuna")
        optuna_frame.grid(row=16, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Botones de control de optimización Optuna
        optuna_buttons_frame = ttk.Frame(optuna_frame)
        optuna_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_optuna_button = ttk.Button(optuna_buttons_frame, text="Iniciar Optimización Optuna", 
                                             command=self.start_optuna_optimization)
        self.start_optuna_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_optuna_button = ttk.Button(optuna_buttons_frame, text="Detener Optimización Optuna", 
                                            command=self.stop_optuna_optimization, state=tk.DISABLED)
        self.stop_optuna_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_optuna_params_button = ttk.Button(optuna_buttons_frame, text="Cargar Mejores Parámetros Optuna", 
                                                   command=self.load_optuna_params)
        self.load_optuna_params_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Nuevo botón para aplicar parámetros a config.yaml
        self.apply_optuna_to_config_button = ttk.Button(optuna_buttons_frame, text="Aplicar a Config.yaml", 
                                                       command=self.apply_optuna_to_config)
        self.apply_optuna_to_config_button.pack(side=tk.LEFT)
        
        # Estado de optimización Optuna (desde configuración)
        self.optuna_status_var = tk.StringVar(value=status_messages['optuna_inactive'])
        optuna_status_label = ttk.Label(optuna_frame, textvariable=self.optuna_status_var, 
                                       style='Info.TLabel')
        optuna_status_label.pack(anchor='w', padx=10, pady=2)
        
        # Progreso de optimización Optuna
        self.optuna_progress_var = tk.DoubleVar()
        self.optuna_progress_bar = ttk.Progressbar(optuna_frame, variable=self.optuna_progress_var, 
                                                  maximum=100)
        self.optuna_progress_bar.pack(fill=tk.X, padx=10, pady=2)

    def create_custom_functions_section(self, parent):
        """Crear sección de funciones personalizadas"""
        custom_frame = ttk.LabelFrame(parent, text="Funciones Personalizadas")
        custom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Marco principal organizado horizontalmente
        main_frame = ttk.Frame(custom_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna izquierda - Coordenadas y Matrices
        left_column = ttk.Frame(main_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Coordenadas iniciales
        coord_frame = ttk.LabelFrame(left_column, text="Coordenadas Iniciales")
        coord_frame.pack(fill=tk.X, pady=(0, 10))
        
        coord_inner = ttk.Frame(coord_frame)
        coord_inner.pack(padx=10, pady=10)
        
        # Coordenadas desde configuración
        defaults_config = self.config_manager.get_defaults_config()
        
        ttk.Label(coord_inner, text="x0:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.x0_var = tk.StringVar(value=str(defaults_config['coordinates']['x0']))
        ttk.Entry(coord_inner, textvariable=self.x0_var, width=8).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(coord_inner, text="y0:").grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.y0_var = tk.StringVar(value=str(defaults_config['coordinates']['y0']))
        ttk.Entry(coord_inner, textvariable=self.y0_var, width=8).grid(row=0, column=3)
        
        # Matrices lado a lado
        matrices_frame = ttk.Frame(left_column)
        matrices_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Matriz A1
        a1_frame = ttk.LabelFrame(matrices_frame, text="Matriz A1 (Modo 0)")
        a1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        a1_inner = ttk.Frame(a1_frame)
        a1_inner.pack(padx=10, pady=10)
        
        # Matriz A1 desde configuración
        default_A1 = defaults_config['matrices']['A1']
        self.a1_vars = []
        for i in range(2):
            row_vars = []
            for j in range(2):
                var = tk.StringVar(value=str(default_A1[i][j]))
                ttk.Entry(a1_inner, textvariable=var, width=8).grid(row=i, column=j, padx=2, pady=2)
                row_vars.append(var)
            self.a1_vars.append(row_vars)
        
        # Matriz A2
        a2_frame = ttk.LabelFrame(matrices_frame, text="Matriz A2 (Modo 1)")
        a2_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        a2_inner = ttk.Frame(a2_frame)
        a2_inner.pack(padx=10, pady=10)
        
        # Matriz A2 desde configuración
        default_A2 = defaults_config['matrices']['A2']
        self.a2_vars = []
        for i in range(2):
            row_vars = []
            for j in range(2):
                var = tk.StringVar(value=str(default_A2[i][j]))
                ttk.Entry(a2_inner, textvariable=var, width=8).grid(row=i, column=j, padx=2, pady=2)
                row_vars.append(var)
            self.a2_vars.append(row_vars)
        
        # Columna derecha - Función de recompensa y controles
        right_column = ttk.Frame(main_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Función de recompensa
        reward_frame = ttk.LabelFrame(right_column, text="Función de Recompensa")
        reward_frame.pack(fill=tk.X, pady=(0, 10))
        
        reward_inner = ttk.Frame(reward_frame)
        reward_inner.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(reward_inner, text="Expresión (usar x, y, x0, y0):").pack(anchor='w')
        
        self.reward_expr_var = tk.StringVar(value=defaults_config['matrices']['reward_function'])
        reward_entry = ttk.Entry(reward_inner, textvariable=self.reward_expr_var, width=50)
        reward_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Checkbox para función inteligente de recompensa (ACTIVADO por defecto para mejor resultado)
        self.smart_reward_var = tk.BooleanVar(value=True)
        smart_reward_cb = ttk.Checkbutton(reward_inner, 
                                        text="🧠 Usar Recompensa Inteligente (Mode-Aware)", 
                                        variable=self.smart_reward_var,
                                        command=self.toggle_smart_reward)
        smart_reward_cb.pack(anchor='w', pady=(5, 0))
        
        # Etiqueta explicativa para recompensa inteligente
        smart_reward_info = ttk.Label(reward_inner, text="Enseña automáticamente cuándo usar cada modo", 
                                     font=('Arial', 8), foreground='gray')
        smart_reward_info.pack(anchor='w', padx=(20, 0))
        
        # Selector de optimización de recompensa
        ttk.Label(reward_inner, text="Optimización de recompensa:").pack(anchor='w', pady=(10, 0))
        self.reward_optimization_var = tk.StringVar(value=defaults_config['matrices']['reward_optimization'])
        reward_optimization_frame = ttk.Frame(reward_inner)
        reward_optimization_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Radiobutton(reward_optimization_frame, text="Minimizar", 
                       variable=self.reward_optimization_var, value="minimizar").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(reward_optimization_frame, text="Maximizar", 
                       variable=self.reward_optimization_var, value="maximizar").pack(side=tk.LEFT)
        
        # Checkbox y botones
        controls_frame = ttk.LabelFrame(right_column, text="Controles")
        controls_frame.pack(fill=tk.X)
        
        controls_inner = ttk.Frame(controls_frame)
        controls_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Checkbox
        self.use_custom_functions_var = tk.BooleanVar(value=self.checkbox_defaults['use_custom_functions'])
        ttk.Checkbutton(controls_inner, text="Usar Funciones Personalizadas", 
                       variable=self.use_custom_functions_var).pack(anchor='w', pady=(0, 10))
        
        # Botones organizados horizontalmente
        button_frame = ttk.Frame(controls_inner)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Valores Por Defecto", 
                  command=self.load_default_values).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Recargar Config", 
                  command=self.reload_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Auto-recargar", 
                  command=self.toggle_auto_reload).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Mostrar Config", 
                  command=self.show_current_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Probar", 
                  command=self.test_custom_functions).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Guardar", 
                  command=self.save_custom_functions).pack(side=tk.LEFT)

    def load_default_values(self):
        """Cargar valores por defecto desde config.yaml"""
        # Coordenadas (desde config.yaml)
        defaults_config = self.config_manager.get_defaults_config()
        self.x0_var.set(str(defaults_config['coordinates']['x0']))
        self.y0_var.set(str(defaults_config['coordinates']['y0']))
        
        # Matriz A1 (desde config.yaml)
        a1_defaults = defaults_config['matrices']['A1']
        for i in range(2):
            for j in range(2):
                self.a1_vars[i][j].set(str(a1_defaults[i][j]))
        
        # Matriz A2 (desde config.yaml)
        a2_defaults = defaults_config['matrices']['A2']
        for i in range(2):
            for j in range(2):
                self.a2_vars[i][j].set(str(a2_defaults[i][j]))
        
        # **NUEVO**: Función de recompensa desde config.yaml
        self.reward_expr_var.set(defaults_config['matrices']['reward_function'])
        self.reward_optimization_var.set(defaults_config['matrices']['reward_optimization'])
        
        print("✅ Valores por defecto cargados desde config.yaml")

    def reload_config(self):
        """Recargar configuración desde config.yaml"""
        try:
            print("🔄 Recargando configuración desde config.yaml...")
            
            # Recargar el config manager
            self.config_manager.load_config()
            
            # Recargar valores por defecto
            self.load_default_values()
            
            # Recargar configuración de interfaz
            self.interface_config = self.config_manager.get_interface_config()
            self.checkbox_defaults = self.interface_config['checkboxes']
            
            # Actualizar checkboxes
            self.show_rewards_var.set(self.checkbox_defaults['show_rewards'])
            self.show_precision_var.set(self.checkbox_defaults['show_precision'])
            self.show_loss_var.set(self.checkbox_defaults['show_loss'])
            self.reward_normalize_var.set(self.checkbox_defaults['reward_normalize'])
            self.reward_shaping_var.set(self.checkbox_defaults['reward_shaping'])
            self.smart_reward_var.set(self.checkbox_defaults['smart_reward'])
            self.use_custom_functions_var.set(self.checkbox_defaults['use_custom_functions'])
            
            print("✅ Configuración recargada exitosamente")
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error recargando configuración\n" \
                       f"   Error: {e}\n" \
                       f"   Revisa el archivo config.yaml"
            print(error_msg)
            if not self.cli_mode:
                messagebox.showerror("Error", f"Error recargando configuración:\n{e}")

    def toggle_auto_reload(self):
        """Alternar auto-recarga de configuración"""
        if not hasattr(self, 'auto_reload_enabled'):
            self.auto_reload_enabled = False
        
        self.auto_reload_enabled = not self.auto_reload_enabled
        
        if self.auto_reload_enabled:
            print("🔄 Auto-recarga habilitada - Monitoreando config.yaml")
            self.start_config_monitor()
        else:
            print("⏹️ Auto-recarga deshabilitada")
            self.stop_config_monitor()

    def start_config_monitor(self):
        """Iniciar monitoreo del archivo config.yaml"""
        import os
        import time
        import threading
        
        def monitor_config():
            config_path = self.config_manager.config_path
            last_modified = os.path.getmtime(config_path)
            
            while self.auto_reload_enabled:
                try:
                    current_modified = os.path.getmtime(config_path)
                    if current_modified > last_modified:
                        print("📝 Cambio detectado en config.yaml - Recargando...")
                        self.root.after(0, self.reload_config)
                        last_modified = current_modified
                    
                    time.sleep(1)  # Verificar cada segundo
                except Exception as e:
                    print(f"❌ Error en monitoreo: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_config, daemon=True)
        self.monitor_thread.start()

    def stop_config_monitor(self):
        """Detener monitoreo del archivo config.yaml"""
        if hasattr(self, 'auto_reload_enabled'):
            self.auto_reload_enabled = False

    def show_current_config(self):
        """Mostrar configuración actual en una ventana emergente"""
        try:
            import tkinter as tk
            from tkinter import scrolledtext
            
            # Crear ventana emergente
            config_window = tk.Toplevel(self.root)
            config_window.title("Configuración Actual")
            config_window.geometry("600x400")
            
            # Área de texto con scroll
            text_area = scrolledtext.ScrolledText(config_window, wrap=tk.WORD, width=70, height=20)
            text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Obtener configuración actual
            defaults_config = self.config_manager.get_defaults_config()
            
            # Mostrar información de configuración
            config_info = f"""CONFIGURACIÓN ACTUAL (config.yaml)

📍 COORDENADAS INICIALES:
   x0: {defaults_config['coordinates']['x0']}
   y0: {defaults_config['coordinates']['y0']}

🔄 MATRICES DE TRANSFORMACIÓN:
   A1 (Modo 0): {defaults_config['matrices']['A1']}
   A2 (Modo 1): {defaults_config['matrices']['A2']}

💰 FUNCIÓN DE RECOMPENSA:
   Expresión: {defaults_config['matrices']['reward_function']}
   Optimización: {defaults_config['matrices']['reward_optimization']}

📁 ARCHIVO DE CONFIGURACIÓN:
   {self.config_manager.config_path}

🔄 ESTADO DE AUTO-RECARGA:
   {'Habilitada' if hasattr(self, 'auto_reload_enabled') and self.auto_reload_enabled else 'Deshabilitada'}

💡 CONSEJO: Modifica el archivo config.yaml y usa "Recargar Config" 
   o habilita "Auto-recargar" para ver los cambios automáticamente.
"""
            
            text_area.insert(tk.END, config_info)
            text_area.config(state=tk.DISABLED)  # Solo lectura
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error mostrando configuración\n" \
                       f"   Error: {e}"
            print(error_msg)
            if not self.cli_mode:
                messagebox.showerror("Error", f"Error mostrando configuración:\n{e}")

    def test_custom_functions(self):
        """Probar las funciones personalizadas"""
        try:
            # Obtener valores
            x0 = float(self.x0_var.get())
            y0 = float(self.y0_var.get())
            
            # Construir matrices
            A1 = np.array([[float(self.a1_vars[i][j].get()) for j in range(2)] for i in range(2)])
            A2 = np.array([[float(self.a2_vars[i][j].get()) for j in range(2)] for i in range(2)])
            
            # Probar transformaciones
            coord_vector = np.array([[x0], [y0]])
            x1 = A1 @ coord_vector
            x2 = A2 @ coord_vector
            
            print("🧪 PRUEBA DE FUNCIONES PERSONALIZADAS:")
            print(f"📍 Coordenadas iniciales: ({x0}, {y0})")
            print(f"🔄 Matriz A1:\n{A1}")
            print(f"   Transformación x1: {x1.flatten()}")
            print(f"🔄 Matriz A2:\n{A2}")
            print(f"   Transformación x2: {x2.flatten()}")
            
            # Probar función de recompensa
            reward_expr = self.reward_expr_var.get()
            x, y = 3, 4  # Punto de prueba
            
            # Crear namespace seguro para eval
            safe_dict = {
                'x': x, 'y': y, 'x0': x0, 'y0': y0,
                'np': np, 'abs': abs, 'sqrt': np.sqrt,
                '__builtins__': {}
            }
            
            reward_value = eval(reward_expr, safe_dict)
            print(f"💰 Función de recompensa para punto ({x}, {y}): {reward_value}")
            print("✅ Todas las funciones funcionan correctamente")
            
        except Exception as e:
            print(f"❌ Error en las funciones personalizadas: {e}")

    def save_custom_functions(self):
        """Guardar funciones personalizadas en config.yaml"""
        try:
            print("💾 Guardando funciones personalizadas en config.yaml...")
            
            # Obtener configuración actual
            config_data = self.config_manager.config.copy()
            
            # Actualizar coordenadas
            config_data['defaults']['coordinates']['x0'] = float(self.x0_var.get())
            config_data['defaults']['coordinates']['y0'] = float(self.y0_var.get())
            
            # Actualizar matrices
            A1 = [[float(self.a1_vars[i][j].get()) for j in range(2)] for i in range(2)]
            A2 = [[float(self.a2_vars[i][j].get()) for j in range(2)] for i in range(2)]
            config_data['defaults']['matrices']['A1'] = A1
            config_data['defaults']['matrices']['A2'] = A2
            
            # Actualizar función de recompensa
            config_data['defaults']['matrices']['reward_function'] = self.reward_expr_var.get()
            config_data['defaults']['matrices']['reward_optimization'] = self.reward_optimization_var.get()
            
            # Guardar en config.yaml
            self.config_manager.save_config(config_data, backup=True)
            
            print("✅ Funciones personalizadas guardadas en config.yaml")
            if not self.cli_mode:
                messagebox.showinfo("Guardado", "Funciones personalizadas guardadas en config.yaml")
                
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error guardando funciones personalizadas\n" \
                       f"   Error: {e}\n" \
                       f"   Revisa los valores ingresados"
            print(error_msg)
            if not self.cli_mode:
                messagebox.showerror("Error", f"Error guardando funciones:\n{e}")

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
    
    def create_plot_controls(self, parent):
        """Crear controles para seleccionar los gráficos a mostrar."""
        plot_controls_frame = ttk.LabelFrame(parent, text="Controles de Gráficos", padding=10)
        plot_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))

        # El comando solo actualiza la GUI, no el pop-up
        ttk.Checkbutton(plot_controls_frame, text="Recompensas", variable=self.show_rewards_var, command=self.update_gui_plot).pack(anchor='w')
        ttk.Checkbutton(plot_controls_frame, text="Precisión", variable=self.show_precision_var, command=self.update_gui_plot).pack(anchor='w')
        ttk.Checkbutton(plot_controls_frame, text="Pérdida", variable=self.show_loss_var, command=self.update_gui_plot).pack(anchor='w')

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
        """Crear la sección de gráficos."""
        right_pane = ttk.Frame(parent)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plots_frame = ttk.LabelFrame(right_pane, text="Gráficos de Resultados")
        plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        if MATPLOTLIB_AVAILABLE:
            # Crear gráficos con matplotlib
            self.fig = plt.Figure(figsize=(6, 5), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            # Mostrar mensaje cuando matplotlib no está disponible
            no_plot_label = ttk.Label(plots_frame, text="Gráficos no disponibles\n(matplotlib requerido)", 
                                    font=('Arial', 12), foreground='gray')
            no_plot_label.pack(expand=True)

        self.create_plot_controls(plots_frame)

    def redirect_output(self):
        """Redirigir la salida de print al widget de texto"""
        self.redirect = RedirectText(self.output_text)
        sys.stdout = self.redirect
    
    def start_training(self):
        """Iniciar entrenamiento en un hilo separado"""
        print("DEBUG: Botón 'Iniciar Entrenamiento' presionado")
        if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.is_alive():
            print("DEBUG: Entrenamiento ya en progreso")
            if not self.cli_mode:
                messagebox.showwarning("Advertencia", "El entrenamiento ya está en progreso")
            return
        
        # Obtener parámetros
        params = self.get_training_parameters()
        print("DEBUG: Parámetros obtenidos:", params)
        
        # **CORREGIDO**: SIEMPRE leer matrices del GUI independientemente de funciones personalizadas
        try:
            # Matrices ya incluidas en get_training_parameters
            
            # Verificar si usar funciones personalizadas
            if self.use_custom_functions_var.get():
                # Construir funciones personalizadas desde los campos de la interfaz
                x0 = float(self.x0_var.get())
                y0 = float(self.y0_var.get())
                
                # Función de recompensa
                reward_expr = self.reward_expr_var.get()
                
                # Almacenar en los parámetros
                params['custom_functions'] = {
                    'x0': x0,
                    'y0': y0,
                    'A1': A1,
                    'A2': A2,
                    'reward_expr': reward_expr
                }
                print("Usando configuración personalizada")
            else:
                print("Usando matrices del GUI con funciones por defecto")
                
        except Exception as e:
            print(f"Error al procesar matrices del GUI: {e}")
            if not self.cli_mode:
                messagebox.showerror("Error", f"Error en matrices del GUI: {e}")
            return
        
        # Actualizar interfaz (solo en modo GUI)
        if not self.cli_mode:
            self.train_button.config(state=tk.DISABLED)
            self.status_label.config(text="Entrenando...")
            self.progress_var.set(0)
        
        # Iniciar entrenamiento en hilo separado
        self.training_thread = threading.Thread(target=self.run_training, args=(params,))
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def get_training_parameters(self):
        """Obtener parámetros de la interfaz (con conversión de tipos segura)"""
        try:
            # Construir matrices A1 y A2 desde los campos de la interfaz
            A1 = [[float(self.a1_vars[i][j].get()) for j in range(2)] for i in range(2)]
            A2 = [[float(self.a2_vars[i][j].get()) for j in range(2)] for i in range(2)]
            
            params = {
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
                'max_steps': int(self.max_steps_var.get()),  # Convertir a int
                'buffer_capacity': int(self.buffer_capacity_var.get()),  # Convertir a int
                'alpha': float(self.alpha_var.get()),  # Convertir a float
                'beta': float(self.beta_var.get()),  # Convertir a float
                'supervised_episodes': self.supervised_episodes_var.get(),  # Ya es IntVar
                'reward_normalize': self.reward_normalize_var.get(),
                'reward_shaping': self.reward_shaping_var.get(),
                'reward_optimization': self.reward_optimization_var.get(),
                'gui_reward_function': 'mode_aware_reward' if self.smart_reward_var.get() else self.reward_expr_var.get(),
                'gui_matrices': {
                    'A1': A1,
                    'A2': A2
                }
            }
            # **DEBUG**: Imprimir la función de recompensa que se está enviando
            print(f"DEBUG: Función de recompensa de GUI: '{params['gui_reward_function']}'")
            return params
        except ValueError as e:
            error_msg = f"❌ ERROR EN PARÁMETROS DE GUI:\n" \
                       f"   Error de conversión: {e}\n" \
                       f"   Revisa que todos los campos tengan valores válidos"
            print(error_msg)
            raise RuntimeError(f"Parámetros de GUI inválidos: {e}")
    
    def run_training(self, params):
        """Ejecutar entrenamiento con parámetros dados"""
        # Importar módulo de entrenamiento
        # from training_manager import TrainingManager # This line is now at the top
        
        try:
            # Crear manager de entrenamiento
            training_manager = TrainingManager()
            
            # Ejecutar entrenamiento
            self.hnaf_model, self.training_results = training_manager.train_hnaf(params)
            
            # Actualizar interfaz (solo en modo GUI)
            if not self.cli_mode:
                self.root.after(0, self.training_completed)
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {str(e)}")
            if not self.cli_mode:
                self.root.after(0, self.training_error, str(e))
    
    def training_completed(self):
        """Llamado cuando el entrenamiento se completa"""
        if not self.cli_mode:
            self.train_button.config(state=tk.NORMAL)
            self.eval_button.config(state=tk.NORMAL)
            self.verify_button.config(state=tk.NORMAL)
            self.status_label.config(text="Entrenamiento completado")
            self.progress_var.set(100)
            
            # Lógica de actualización de gráficos refactorizada
            self.update_gui_plot()
            self.show_plots_in_popup()

            self.train_button['state'] = 'normal'
            self.verify_button['state'] = 'normal'

    def training_error(self, error_msg):
        """Llamado cuando hay un error en el entrenamiento"""
        if not self.cli_mode:
            self.train_button.config(state=tk.NORMAL)
            self.status_label.config(text="Error en entrenamiento")
        if not self.cli_mode:
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
        
        # Evaluación simple
        if hasattr(self.hnaf_model, 'verify_hnaf'):
            self.hnaf_model.verify_hnaf()
        
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
    
    def update_gui_plot(self):
        """Actualiza solo el gráfico incrustado en la GUI principal."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Verificar si hay algo que mostrar
        show_any = self.show_rewards_var.get() or self.show_precision_var.get() or self.show_loss_var.get()
        
        if not show_any or self.training_results is None:
            # Si no hay nada seleccionado o no hay resultados, limpiar el gráfico
            if hasattr(self, 'fig') and hasattr(self, 'canvas'):
                self.fig.clear()
                self.canvas.draw()
            return

        # Gráfico simple
        if hasattr(self, 'fig') and hasattr(self, 'canvas'):
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            if 'episode_rewards' in self.training_results:
                rewards = self.training_results['episode_rewards']
                ax.plot(rewards)
                ax.set_title('Recompensas por Episodio')
            self.canvas.draw()

    def show_plots_in_popup(self):
        """Muestra los gráficos en una ventana emergente de alta resolución."""
        if not self.training_results or not MATPLOTLIB_AVAILABLE:
            if not MATPLOTLIB_AVAILABLE:
                if not self.cli_mode:
                    messagebox.showwarning("Gráficos no disponibles", 
                                         "Matplotlib no está instalado. No se pueden mostrar gráficos.")
            return

        if not self.cli_mode:
            popup = tk.Toplevel(self.root)
            popup.title("Gráficos de Resultados")
            popup.geometry("800x600")

            # Gráfico simple
            fig_popup = plt.Figure(figsize=(8, 6), dpi=100)
            canvas_popup = FigureCanvasTkAgg(fig_popup, master=popup)
            canvas_popup.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            ax = fig_popup.add_subplot(111)
            if 'episode_rewards' in self.training_results:
                rewards = self.training_results['episode_rewards']
                ax.plot(rewards)
                ax.set_title('Recompensas por Episodio')
            canvas_popup.draw()

    def run_training_thread(self):
        """Ejecuta el entrenamiento en un hilo separado."""
        # ... existing code ...

    def clear_output(self):
        """Limpiar la salida de texto"""
        print("DEBUG: Botón 'Limpiar Salida' presionado")
        self.output_text.delete(1.0, tk.END)
        print("DEBUG: Salida limpiada")
    
    def start_gemini_optimization(self):
        """Iniciar optimización automática con Gemini"""
        try:
            print("Iniciando optimización automática con Gemini...")
            
            # Importar módulo de optimización
            from final_app.auto_optimizer import AutoOptimizer
            from final_app.training_manager import TrainingManager
            
            # Crear optimizador
            self.auto_optimizer = AutoOptimizer()
            self.training_manager = TrainingManager()
            
            # Actualizar UI
            self.start_optimization_button.config(state=tk.DISABLED)
            self.stop_optimization_button.config(state=tk.NORMAL)
            self.optimization_status_var.set("Optimización: Ejecutando...")
            self.optimization_progress_var.set(0)
            
            # Callback para actualizar progreso
            def update_progress(iteration, current_score, best_score):
                progress = (iteration / 50) * 100  # 50 iteraciones máximo
                self.optimization_progress_var.set(progress)
                self.optimization_status_var.set(f"Optimización: Iteración {iteration}/50 - Score: {current_score:.4f} (Mejor: {best_score:.4f})")
                self.root.update_idletasks()
            
            # Iniciar optimización en thread separado
            self.optimization_thread = self.auto_optimizer.start_optimization(
                self.training_manager, 
                callback=update_progress
            )
            
            print("Optimización iniciada en segundo plano")
            
        except Exception as e:
            print(f"❌ Error iniciando optimización: {e}")
            self.optimization_status_var.set("Optimización: Error")
    
    def stop_gemini_optimization(self):
        """Detener optimización automática"""
        try:
            if hasattr(self, 'auto_optimizer'):
                self.auto_optimizer.stop_optimization()
            
            # Actualizar UI
            self.start_optimization_button.config(state=tk.NORMAL)
            self.stop_optimization_button.config(state=tk.DISABLED)
            self.optimization_status_var.set("Optimización: Detenida")
            
            print("Optimización detenida")
            
        except Exception as e:
            print(f"❌ Error deteniendo optimización: {e}")
    
    def load_best_params(self):
        """Cargar mejores parámetros encontrados por Gemini"""
        try:
            from final_app.auto_optimizer import AutoOptimizer
            
            optimizer = AutoOptimizer()
            best_params = optimizer.get_best_params()
            
            if best_params:
                # Aplicar mejores parámetros a la UI (con fallbacks desde configuración)
                network_defaults = self.config_manager.get_network_defaults()
                training_defaults = self.config_manager.get_training_defaults()
                
                self.hidden_dim_var.set(best_params.get('hidden_dim', network_defaults['hidden_dim']))
                self.num_layers_var.set(best_params.get('num_layers', network_defaults['num_layers']))
                self.learning_rate_var.set(best_params.get('lr', training_defaults['learning_rate']))
                self.batch_size_var.set(best_params.get('batch_size', training_defaults['batch_size']))
                self.initial_epsilon_var.set(best_params.get('initial_epsilon', training_defaults['initial_epsilon']))
                self.final_epsilon_var.set(best_params.get('final_epsilon', training_defaults['final_epsilon']))
                self.max_steps_var.set(str(best_params.get('max_steps', training_defaults['max_steps'])))
                self.buffer_capacity_var.set(str(best_params.get('buffer_capacity', training_defaults['buffer_capacity'])))
                self.alpha_var.set(str(best_params.get('alpha', training_defaults['alpha'])))
                self.beta_var.set(str(best_params.get('beta', training_defaults['beta'])))
                
                print("Mejores parámetros cargados desde optimización automática")
                print(f"   Score: {optimizer.best_score:.4f}")
                print(f"   Parámetros: {best_params}")
                
                messagebox.showinfo("Mejores Parámetros", 
                                  f"Parámetros optimizados cargados\nScore: {optimizer.best_score:.4f}")
            else:
                print("No se encontraron mejores parámetros")
                messagebox.showwarning("Sin Parámetros", 
                                     "No se encontraron parámetros optimizados")
                
        except Exception as e:
            print(f"❌ Error cargando mejores parámetros: {e}")
            messagebox.showerror("Error", f"Error cargando mejores parámetros: {e}")
    
    def start_optuna_optimization(self):
        """Iniciar optimización automática con Optuna"""
        try:
            print("Iniciando optimización automática con Optuna...")
            
            # Importar módulo de optimización Optuna
            from final_app.optuna_optimizer import OptunaOptimizer
            
            # Crear optimizador Optuna
            self.optuna_optimizer = OptunaOptimizer()
            
            # FORZAR configuración ilimitada (para optimización nocturna)
            self.optuna_optimizer.max_iterations = None
            self.optuna_optimizer.timeout_minutes = None
            # Usar configuración en lugar de hardcodear
            hardcode_config = self.config_manager.get_hardcode_elimination_config()
            self.optuna_optimizer.evaluation_episodes = hardcode_config['optuna']['default_evaluation_episodes']
            print(f"🔄 Configuración Optuna forzada: max_iterations={self.optuna_optimizer.max_iterations}, timeout={self.optuna_optimizer.timeout_minutes}, episodes={self.optuna_optimizer.evaluation_episodes}")
            
            # Actualizar UI
            self.start_optuna_button.config(state=tk.DISABLED)
            self.stop_optuna_button.config(state=tk.NORMAL)
            self.optuna_status_var.set("Optimización Optuna: Ejecutando (SIN LÍMITES)...")
            self.optuna_progress_var.set(0)
            
            # Callback para actualizar progreso (SIN límite de 50)
            def update_optuna_progress(iteration, current_score, best_score):
                # Sin límite máximo - solo mostrar progreso continuo
                self.optuna_progress_var.set(min(iteration * 2, 100))  # Progreso visual
                self.optuna_status_var.set(f"Optimización Optuna: Trial {iteration} - Score: {current_score:.4f} (Mejor: {best_score:.4f}) [SIN LÍMITES]")
                self.root.update_idletasks()
            
            # Iniciar optimización en thread separado
            self.optuna_optimizer.start_optimization(progress_callback=update_optuna_progress)
            
            print("Optimización Optuna iniciada en segundo plano")
            
        except Exception as e:
            print(f"Error iniciando optimización Optuna: {e}")
            self.optuna_status_var.set("Optimización Optuna: Error")
    
    def stop_optuna_optimization(self):
        """Detener optimización automática con Optuna"""
        try:
            if hasattr(self, 'optuna_optimizer'):
                self.optuna_optimizer.stop_optimization()
            
            # Actualizar UI
            self.start_optuna_button.config(state=tk.NORMAL)
            self.stop_optuna_button.config(state=tk.DISABLED)
            self.optuna_status_var.set("Optimización Optuna: Detenida")
            
            print("Optimización Optuna detenida")
            
        except Exception as e:
            print(f"Error deteniendo optimización Optuna: {e}")
    
    def load_optuna_params(self):
        """Cargar mejores parámetros encontrados por Optuna"""
        try:
            from final_app.optuna_optimizer import OptunaOptimizer
            
            optimizer = OptunaOptimizer()
            best_params = optimizer.get_best_params()
            
            if best_params:
                # Aplicar mejores parámetros a la UI (con fallbacks desde configuración)
                network_defaults = self.config_manager.get_network_defaults()
                training_defaults = self.config_manager.get_training_defaults()
                
                self.hidden_dim_var.set(best_params.get('hidden_dim', network_defaults['hidden_dim']))
                self.num_layers_var.set(best_params.get('num_layers', network_defaults['num_layers']))
                self.learning_rate_var.set(best_params.get('lr', training_defaults['learning_rate']))
                self.batch_size_var.set(best_params.get('batch_size', training_defaults['batch_size']))
                self.initial_epsilon_var.set(best_params.get('initial_epsilon', training_defaults['initial_epsilon']))
                self.final_epsilon_var.set(best_params.get('final_epsilon', training_defaults['final_epsilon']))
                self.max_steps_var.set(str(best_params.get('max_steps', training_defaults['max_steps'])))
                self.buffer_capacity_var.set(str(best_params.get('buffer_capacity', training_defaults['buffer_capacity'])))
                self.alpha_var.set(str(best_params.get('alpha', training_defaults['alpha'])))
                self.beta_var.set(str(best_params.get('beta', training_defaults['beta'])))
                
                print("Mejores parámetros de Optuna aplicados a la interfaz")
            else:
                print("No se encontraron parámetros optimizados de Optuna")
                
        except Exception as e:
            print(f"Error cargando mejores parámetros de Optuna: {e}")
    
    def apply_optuna_to_config(self):
        """Aplicar mejores parámetros de Optuna al config.yaml"""
        try:
            from final_app.optuna_optimizer import OptunaOptimizer
            
            # Crear instancia del optimizador
            optimizer = OptunaOptimizer()
            
            # Verificar que hay parámetros optimizados
            if not optimizer.best_params:
                messagebox.showwarning("Sin Parámetros", 
                                     "No hay parámetros optimizados de Optuna para aplicar.\n"
                                     "Ejecuta primero la optimización de Optuna.")
                return
            
            # Confirmar con el usuario
            response = messagebox.askyesno(
                "Confirmar Actualización",
                f"¿Aplicar los mejores parámetros de Optuna al config.yaml?\n\n"
                f"Score: {optimizer.best_score:.4f}\n"
                f"Parámetros: {list(optimizer.best_params.keys())}\n\n"
                f"Se creará un backup automático del config.yaml actual."
            )
            
            if not response:
                return
            
            # Aplicar parámetros al config.yaml
            print("🔄 Aplicando mejores parámetros de Optuna a config.yaml...")
            success = optimizer.update_config_yaml()
            
            if success:
                # Mostrar éxito
                messagebox.showinfo(
                    "Config.yaml Actualizado",
                    f"✅ Config.yaml actualizado exitosamente con mejores parámetros de Optuna!\n\n"
                    f"📊 Score: {optimizer.best_score:.4f}\n"
                    f"🔧 Parámetros actualizados: {len(optimizer.best_params)}\n"
                    f"📁 Backup creado automáticamente\n\n"
                    f"⚠️ Reinicia la aplicación para usar los nuevos valores por defecto."
                )
                
                # Opcional: Cargar también los parámetros en la GUI actual
                confirm_load = messagebox.askyesno(
                    "Cargar en GUI",
                    "¿También cargar estos parámetros en la GUI actual?"
                )
                
                if confirm_load:
                    self.load_optuna_params()
                    
            else:
                # Mostrar error
                messagebox.showerror(
                    "Error",
                    "❌ Error actualizando config.yaml.\n"
                    "Los parámetros siguen disponibles en el archivo JSON.\n"
                    "Revisa la consola para más detalles."
                )
                
        except Exception as e:
            print(f"❌ Error aplicando parámetros de Optuna a config.yaml: {e}")
            messagebox.showerror("Error", f"Error aplicando parámetros: {e}")

    def on_closing(self):
        """Maneja el evento de cierre de la ventana."""
        # Detener optimización Gemini si está ejecutándose
        if hasattr(self, 'auto_optimizer') and self.auto_optimizer.is_running:
            self.auto_optimizer.stop_optimization()
        
        # Detener optimización Optuna si está ejecutándose
        if hasattr(self, 'optuna_optimizer') and self.optuna_optimizer.is_running:
            self.optuna_optimizer.stop_optimization()
        
        # Detener monitoreo de configuración si está activo
        if hasattr(self, 'auto_reload_enabled') and self.auto_reload_enabled:
            self.stop_config_monitor()
        
        print("DEBUG: Cerrando aplicación")
        sys.stdout = sys.__stdout__  # Restaurar stdout
        self.root.destroy()
    
    def toggle_smart_reward(self):
        """Alternar entre función tradicional y función inteligente"""
        if self.smart_reward_var.get():
            print("🧠 Activada: Función de recompensa inteligente (mode-aware)")
            # Mensaje informativo
            print("   ✅ La función enseñará al agente qué modo elegir")
            print("   ✅ Penalizará elección incorrecta de modo")
            print("   ✅ Usará matrices dinámicas desde GUI")
        else:
            print("📊 Activada: Función de recompensa tradicional")
            print(f"   📝 Usando expresión: {self.reward_expr_var.get()}")
    
    def _init_cli_variables(self):
        """Inicializar variables necesarias para modo CLI"""
        # Cargar valores por defecto desde configuración
        network_defaults = self.config_manager.get_network_defaults()
        training_defaults = self.config_manager.get_training_defaults()
        
        # Variables de red neuronal
        self.state_dim_var = tk.IntVar(value=network_defaults['state_dim'])
        self.action_dim_var = tk.IntVar(value=network_defaults['action_dim'])
        self.num_modes_var = tk.IntVar(value=network_defaults['num_modes'])
        self.hidden_dim_var = tk.IntVar(value=network_defaults['hidden_dim'])
        self.num_layers_var = tk.IntVar(value=network_defaults['num_layers'])
        
        # Variables de entrenamiento
        self.learning_rate_var = tk.DoubleVar(value=training_defaults['learning_rate'])
        self.tau_var = tk.DoubleVar(value=training_defaults['tau'])
        self.gamma_var = tk.DoubleVar(value=training_defaults['gamma'])
        self.num_episodes_var = tk.IntVar(value=training_defaults['num_episodes'])
        self.batch_size_var = tk.IntVar(value=training_defaults['batch_size'])
        self.initial_epsilon_var = tk.DoubleVar(value=training_defaults['initial_epsilon'])
        self.final_epsilon_var = tk.DoubleVar(value=training_defaults['final_epsilon'])
        self.max_steps_var = tk.StringVar(value=str(training_defaults['max_steps']))
        self.buffer_capacity_var = tk.StringVar(value=str(training_defaults['buffer_capacity']))
        self.alpha_var = tk.StringVar(value=str(training_defaults['alpha']))
        self.beta_var = tk.StringVar(value=str(training_defaults['beta']))
        self.supervised_episodes_var = tk.IntVar(value=training_defaults['supervised_episodes'])
        
        # Variables de checkboxes
        self.reward_normalize_var = tk.BooleanVar(value=self.checkbox_defaults['reward_normalize'])
        self.reward_shaping_var = tk.BooleanVar(value=self.checkbox_defaults['reward_shaping'])
        self.smart_reward_var = tk.BooleanVar(value=self.checkbox_defaults['smart_reward'])
        
        # Variables de coordenadas
        self.x0_var = tk.StringVar(value=str(self.defaults_config['coordinates']['x0']))
        self.y0_var = tk.StringVar(value=str(self.defaults_config['coordinates']['y0']))
        
        # Variables de matrices A1 y A2
        self.a1_vars = []
        self.a2_vars = []
        default_A1 = self.defaults_config['matrices']['A1']
        default_A2 = self.defaults_config['matrices']['A2']
        
        for i in range(2):
            row1 = []
            row2 = []
            for j in range(2):
                var1 = tk.StringVar(value=str(default_A1[i][j]))
                var2 = tk.StringVar(value=str(default_A2[i][j]))
                row1.append(var1)
                row2.append(var2)
            self.a1_vars.append(row1)
            self.a2_vars.append(row2)
        
        # Variables de función de recompensa
        self.reward_expr_var = tk.StringVar(value=self.defaults_config['matrices']['reward_function'])
        self.reward_optimization_var = tk.StringVar(value=self.defaults_config['matrices']['reward_optimization'])
        
        # Variables de control
        self.use_custom_functions_var = tk.BooleanVar(value=self.checkbox_defaults['use_custom_functions'])
        
        print("✅ Variables CLI inicializadas con valores por defecto")
    
    def run(self):
        """Ejecutar la GUI"""
        if not self.cli_mode:
            self.root.mainloop()
    
    def run_training_cli(self):
        """Ejecutar entrenamiento en modo CLI (exactamente como GUI)"""
        if self.cli_mode:
            print("🎮 EJECUTANDO CÓDIGO IDÉNTICO A LA GUI")
            print("="*50)
            
            # Obtener parámetros exactamente como la GUI
            params = self.get_training_parameters()
            
            print("📋 Parámetros obtenidos (idénticos a GUI):")
            for key, value in params.items():
                if key != 'gui_matrices':
                    print(f"   {key}: {value}")
            print(f"   gui_matrices: A1={params['gui_matrices']['A1']}, A2={params['gui_matrices']['A2']}")
            
            # Ejecutar entrenamiento exactamente como la GUI
            print("\n🚀 Iniciando entrenamiento...")
            self.start_training()
            
            # Esperar a que termine (en CLI, corremos síncronamente)
            if self.training_thread:
                self.training_thread.join()
            
            print("✅ Entrenamiento completado")
            return True
        else:
            print("❌ Error: run_training_cli solo funciona en modo CLI")
            return False

def main():
    """Función principal"""
    print("DEBUG: Iniciando aplicación HNAF GUI")
    root = tk.Tk()
    print("DEBUG: Ventana principal creada")
    
    # Crear config_manager
    config_manager = get_config_manager()
    app = HNAFGUI(root=root, config_manager=config_manager)
    print("DEBUG: Interfaz HNAF creada")
    
    # Configurar cierre limpio
    # root.protocol("WM_DELETE_WINDOW", on_closing) # This line is now handled in __init__
    print("DEBUG: Iniciando mainloop")
    root.mainloop()
    print("DEBUG: Aplicación cerrada")

if __name__ == "__main__":
    main() 