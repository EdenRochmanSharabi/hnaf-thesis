#!/usr/bin/env python3
"""
Sistema de gestión de configuración para HNAF
Elimina TODOS los valores hardcodeados del sistema
"""

import yaml
import os
import sys
from typing import Dict, Any, Optional, Union
import copy

class ConfigManager:
    """Gestor centralizado de configuración sin valores hardcodeados"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar el gestor de configuración
        
        Args:
            config_path: Ruta al archivo config.yaml
        """
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self.load_config()
        
    def _find_config_file(self) -> str:
        """Buscar el archivo config.yaml en el proyecto"""
        # Buscar primero en el directorio actual (final_app/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.yaml')
        if os.path.exists(config_path):
            return config_path
        
        # Si no se encuentra, buscar en el directorio padre
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'config.yaml')
        if os.path.exists(config_path):
            return config_path
            
        # Si no se encuentra, crear uno por defecto
        raise FileNotFoundError(
            f"❌ ERROR CRÍTICO: config.yaml no encontrado\n"
            f"   Buscado en:\n"
            f"   - {os.path.join(current_dir, 'config.yaml')}\n"
            f"   - {config_path}\n"
            f"   APLICACIÓN ABORTADA - Crea archivo config.yaml"
        )
    
    def load_config(self) -> None:
        """Cargar configuración desde archivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                
            print(f"✅ Configuración cargada desde: {self.config_path}")
            
        except FileNotFoundError:
            error_msg = f"❌ ERROR CRÍTICO: Archivo config.yaml no encontrado\n" \
                       f"   Ruta: {self.config_path}\n" \
                       f"   APLICACIÓN ABORTADA - Crea el archivo de configuración"
            print(error_msg)
            raise RuntimeError("Archivo de configuración no encontrado")
            
        except yaml.YAMLError as e:
            error_msg = f"❌ ERROR CRÍTICO: config.yaml tiene errores de sintaxis\n" \
                       f"   Error: {e}\n" \
                       f"   APLICACIÓN ABORTADA - Corrige sintaxis YAML"
            print(error_msg)
            raise RuntimeError(f"Error de sintaxis en config.yaml: {e}")
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error inesperado cargando configuración\n" \
                       f"   Error: {e}\n" \
                       f"   APLICACIÓN ABORTADA - Revisa archivo config.yaml"
            print(error_msg)
            raise RuntimeError(f"Error cargando configuración: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración usando notación de punto
        
        Args:
            key_path: Ruta del valor (ej: 'network.defaults.hidden_dim')
            default: Valor por defecto si no se encuentra (SIEMPRE None para fallar en errores)
            
        Returns:
            Valor de configuración
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    # SIEMPRE fallar si no se encuentra el parámetro (sin fallbacks)
                    error_msg = f"❌ ERROR CRÍTICO: Parámetro de configuración no encontrado\n" \
                               f"   Clave: {key_path}\n" \
                               f"   Clave faltante: {key}\n" \
                               f"   APLICACIÓN ABORTADA - Corrige el nombre del parámetro en config.yaml"
                    print(error_msg)
                    raise RuntimeError(f"Parámetro {key_path} no encontrado en configuración")
                    
            return value
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error accediendo configuración\n" \
                       f"   Clave: {key_path}\n" \
                       f"   Error: {e}\n" \
                       f"   APLICACIÓN ABORTADA - Revisa config.yaml"
            print(error_msg)
            raise RuntimeError(f"Error accediendo configuración {key_path}: {e}")
    
    def get_hardcode_elimination_config(self) -> Dict[str, Any]:
        """Obtener configuración para eliminar valores hardcodeados"""
        return self.get('hardcode_elimination')
    
    def get_network_defaults(self) -> Dict[str, Any]:
        """Obtener parámetros por defecto de la red neuronal"""
        return {
            'state_dim': self.get('network.defaults.state_dim'),
            'action_dim': self.get('network.defaults.action_dim'),
            'num_modes': self.get('network.defaults.num_modes'),
            'hidden_dim': self.get('network.defaults.hidden_dim'),
            'num_layers': self.get('network.defaults.num_layers')
        }
    
    def get_training_defaults(self) -> Dict[str, Any]:
        """Obtener parámetros por defecto de entrenamiento"""
        return {
            'learning_rate': self.get('training.defaults.learning_rate'),
            'tau': self.get('training.defaults.tau'),
            'gamma': self.get('training.defaults.gamma'),
            'num_episodes': self.get('training.defaults.num_episodes'),
            'batch_size': self.get('training.defaults.batch_size'),
            'initial_epsilon': self.get('training.defaults.initial_epsilon'),
            'final_epsilon': self.get('training.defaults.final_epsilon'),
            'max_steps': self.get('training.defaults.max_steps'),
            'buffer_capacity': self.get('training.defaults.buffer_capacity'),
            'alpha': self.get('training.defaults.alpha'),
            'beta': self.get('training.defaults.beta'),
            'supervised_episodes': self.get('training.defaults.supervised_episodes')
        }
    
    def get_network_ranges(self) -> Dict[str, Any]:
        """Obtener rangos de parámetros de red para GUI"""
        return {
            'state_dim': self.get('network.ranges.state_dim'),
            'action_dim': self.get('network.ranges.action_dim'),
            'num_modes': self.get('network.ranges.num_modes'),
            'hidden_dim': self.get('network.ranges.hidden_dim'),
            'hidden_dim_increment': self.get('network.ranges.hidden_dim_increment'),
            'num_layers': self.get('network.ranges.num_layers')
        }
    
    def get_training_ranges(self) -> Dict[str, Any]:
        """Obtener rangos de parámetros de entrenamiento para GUI"""
        return self.get('training.ranges')
    
    def get_state_limits(self) -> Dict[str, float]:
        """Obtener límites de clipping de estados (antes hardcodeado -5.0, 5.0)"""
        return {
            'min': self.get('numerical.state_limits.min'),
            'max': self.get('numerical.state_limits.max')
        }
    
    def get_tolerances(self) -> Dict[str, Union[float, list]]:
        """Obtener tolerancias numéricas (antes hardcodeadas)"""
        return self.get('numerical.tolerances')
    
    def get_reward_shaping_config(self) -> Dict[str, Any]:
        """Obtener configuración de reward shaping (antes hardcodeada)"""
        return self.get('reward_shaping')
    
    def get_initialization_config(self) -> Dict[str, Any]:
        """Obtener configuración de inicialización (antes hardcodeada)"""
        return self.get('initialization')
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Obtener configuración de evaluación (antes hardcodeada)"""
        return self.get('training.evaluation')
    
    def get_app_config(self) -> Dict[str, Any]:
        """Obtener configuración de la aplicación"""
        return self.get('app')
    
    def get_defaults_config(self) -> Dict[str, Any]:
        """Obtener valores por defecto (coordenadas, matrices, etc.)"""
        return self.get('defaults')
    
    def get_interface_config(self) -> Dict[str, Any]:
        """Obtener configuración de interfaz"""
        return self.get('interface')
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Obtener parámetros avanzados"""
        return self.get('advanced')
    
    def get_profile_config(self, profile_name: str) -> Dict[str, Any]:
        """
        Obtener configuración de un perfil específico
        
        Args:
            profile_name: Nombre del perfil (beginner, intermediate, expert, research)
            
        Returns:
            Configuración del perfil
        """
        return self.get(f'profiles.{profile_name}')
    
    def apply_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Aplicar un perfil y retornar configuración completa
        
        Args:
            profile_name: Nombre del perfil a aplicar
            
        Returns:
            Configuración con valores del perfil aplicados
        """
        try:
            profile_config = self.get_profile_config(profile_name)
            
            # Crear copia de configuración base
            applied_config = copy.deepcopy(self.config)
            
            # Aplicar valores del perfil
            if 'network' in profile_config:
                applied_config['network']['defaults'].update(profile_config['network'])
            if 'training' in profile_config:
                applied_config['training']['defaults'].update(profile_config['training'])
                
            print(f"✅ Perfil '{profile_name}' aplicado: {profile_config.get('description', '')}")
            return applied_config
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error aplicando perfil\n" \
                       f"   Perfil: {profile_name}\n" \
                       f"   Error: {e}\n" \
                       f"   APLICACIÓN ABORTADA - Revisa perfil en config.yaml"
            print(error_msg)
            raise RuntimeError(f"Error aplicando perfil {profile_name}: {e}")
    
    def validate_config(self) -> bool:
        """
        Validar que la configuración tiene todos los parámetros necesarios
        
        Returns:
            True si la configuración es válida
        """
        required_sections = [
            'app', 'network', 'training', 'numerical', 
            'reward_shaping', 'initialization', 'defaults', 
            'interface', 'advanced'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)
        
        if missing_sections:
            error_msg = f"❌ ERROR CRÍTICO: Secciones faltantes en config.yaml\n" \
                       f"   Faltantes: {missing_sections}\n" \
                       f"   APLICACIÓN ABORTADA - Completa config.yaml"
            print(error_msg)
            raise RuntimeError(f"Secciones faltantes en configuración: {missing_sections}")
        
        return True
    
    def save_config(self, config_data: Dict[str, Any], backup: bool = True) -> None:
        """
        Guardar configuración actualizada
        
        Args:
            config_data: Nueva configuración a guardar
            backup: Si hacer backup del archivo anterior
        """
        try:
            if backup and os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.backup"
                os.rename(self.config_path, backup_path)
                print(f"✅ Backup creado: {backup_path}")
            
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, default_flow_style=False, allow_unicode=True)
            
            self.config = config_data
            print(f"✅ Configuración guardada en: {self.config_path}")
            
        except Exception as e:
            error_msg = f"❌ ERROR CRÍTICO: Error guardando configuración\n" \
                       f"   Error: {e}\n" \
                       f"   GUARDADO ABORTADO - Revisa permisos"
            print(error_msg)
            raise RuntimeError(f"Error guardando configuración: {e}")

# Instancia global del gestor de configuración
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Obtener instancia singleton del gestor de configuración"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(key_path: str, default: Any = None) -> Any:
    """Función de conveniencia para obtener configuración"""
    return get_config_manager().get(key_path, default)

# Funciones de conveniencia para acceso rápido
def get_network_defaults() -> Dict[str, Any]:
    """Obtener parámetros por defecto de red"""
    return get_config_manager().get_network_defaults()

def get_training_defaults() -> Dict[str, Any]:
    """Obtener parámetros por defecto de entrenamiento"""
    return get_config_manager().get_training_defaults()

def get_state_limits() -> Dict[str, float]:
    """Obtener límites de estados"""
    return get_config_manager().get_state_limits()

def get_tolerances() -> Dict[str, Union[float, list]]:
    """Obtener tolerancias numéricas"""
    return get_config_manager().get_tolerances()