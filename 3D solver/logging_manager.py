"""
Sistema de logging mejorado para HNAF
Proporciona logging estructurado con manejo de excepciones apropiado
"""

import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional
from config_manager import get_config_manager

class HNAFLogger:
    """Logger mejorado para HNAF con manejo estructurado de excepciones"""
    
    def __init__(self, name: str = "HNAF"):
        self.name = name
        self.config_manager = get_config_manager()
        self.logging_config = self.config_manager.get('logging')
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configurar el logger con la configuración especificada"""
        logger = logging.getLogger(self.name)
        
        # Limpiar handlers existentes
        logger.handlers.clear()
        
        # Configurar nivel
        level = getattr(logging, self.logging_config['level'].upper())
        logger.setLevel(level)
        
        # Formato
        formatter = logging.Formatter(self.logging_config['format'])
        
        # Handler para consola
        if self.logging_config['console_output']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Handler para archivo
        if self.logging_config['file_output']:
            self._setup_file_handler(logger, formatter)
        
        return logger
    
    def _setup_file_handler(self, logger: logging.Logger, formatter: logging.Formatter):
        """Configurar handler de archivo con rotación"""
        log_file = Path(self.logging_config['file_path'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # RotatingFileHandler para manejo de tamaño
        max_bytes = self.logging_config['max_file_size_mb'] * 1024 * 1024
        backup_count = self.logging_config['backup_count']
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def info(self, message: str, **context):
        """Log información con contexto opcional"""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.info(message)
    
    def warning(self, message: str, **context):
        """Log warning con contexto opcional"""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.warning(message)
    
    def error(self, message: str, **context):
        """Log error con contexto opcional"""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.error(message)
    
    def critical(self, message: str, **context):
        """Log crítico con contexto opcional"""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.critical(message)
    
    def debug(self, message: str, **context):
        """Log debug con contexto opcional"""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.debug(message)
    
    def log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None, 
                     operation: str = "Unknown operation"):
        """
        Log una excepción con contexto completo y traceback
        
        Args:
            exception: La excepción a registrar
            context: Contexto adicional (parámetros, estado, etc.)
            operation: Descripción de la operación que falló
        """
        exc_config = self.logging_config['exception_handling']
        
        # Mensaje base
        error_msg = f"EXCEPCIÓN en {operation}: {type(exception).__name__}: {str(exception)}"
        
        # Añadir contexto si está disponible
        if context and exc_config['include_context']:
            error_msg += f" | Contexto: {context}"
        
        # Log del error
        self.logger.error(error_msg)
        
        # Traceback completo si está habilitado
        if exc_config['log_full_traceback']:
            tb_str = ''.join(traceback.format_tb(exception.__traceback__))
            self.logger.error(f"Traceback completo:\n{tb_str}")
    
    def safe_execute(self, operation_name: str, func, *args, **kwargs):
        """
        Ejecutar una función de forma segura con logging automático de errores
        
        Args:
            operation_name: Nombre descriptivo de la operación
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función o None si hay error
        """
        exc_config = self.logging_config['exception_handling']
        max_attempts = exc_config['auto_recovery_attempts']
        
        for attempt in range(max_attempts + 1):
            try:
                if attempt > 0:
                    self.info(f"Reintentando {operation_name} (intento {attempt + 1}/{max_attempts + 1})")
                
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.info(f"Operación {operation_name} exitosa tras {attempt + 1} intentos")
                
                return result
                
            except Exception as e:
                context = {
                    'operation': operation_name,
                    'attempt': attempt + 1,
                    'max_attempts': max_attempts + 1,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                if attempt < max_attempts:
                    self.warning(f"Error en {operation_name}, reintentando...", **context)
                else:
                    self.log_exception(e, context, operation_name)
                    self.error(f"Operación {operation_name} falló tras {max_attempts + 1} intentos")
                    return None

# Instancia global del logger
_global_logger = None

def get_logger(name: str = "HNAF") -> HNAFLogger:
    """Obtener instancia del logger (singleton)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HNAFLogger(name)
    return _global_logger

def log_info(message: str, **context):
    """Función de conveniencia para logging de info"""
    get_logger().info(message, **context)

def log_warning(message: str, **context):
    """Función de conveniencia para logging de warning"""
    get_logger().warning(message, **context)

def log_error(message: str, **context):
    """Función de conveniencia para logging de error"""
    get_logger().error(message, **context)

def log_exception(exception: Exception, context: Optional[Dict[str, Any]] = None, 
                 operation: str = "Unknown operation"):
    """Función de conveniencia para logging de excepciones"""
    get_logger().log_exception(exception, context, operation)

def safe_execute(operation_name: str, func, *args, **kwargs):
    """Función de conveniencia para ejecución segura"""
    return get_logger().safe_execute(operation_name, func, *args, **kwargs)