#!/usr/bin/env python3
"""
Módulo para guardar y cargar modelos HNAF entrenados
Incluye metadatos de entrenamiento para reproducibilidad
"""

import os
import torch
import json
from datetime import datetime
from config_manager import get_config_manager
from logging_manager import get_logger

class ModelSaver:
    """Gestor para guardar y cargar modelos HNAF con metadatos"""
    
    def __init__(self, save_dir="models"):
        self.save_dir = save_dir
        self.logger = get_logger("ModelSaver")
        self.config_manager = get_config_manager()
        
        # Crear directorio si no existe
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model, training_results=None, filename=None):
        """
        Guardar modelo HNAF con metadatos de entrenamiento
        
        Args:
            model: Instancia de HNAFImproved
            training_results: Resultados del entrenamiento
            filename: Nombre personalizado del archivo
        """
        try:
            # Generar nombre de archivo con timestamp
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hnaf_model_{timestamp}.pt"
            
            filepath = os.path.join(self.save_dir, filename)
            
            # Preparar checkpoint
            checkpoint = {
                'model_state_dict': model.model.state_dict(),
                'target_model_state_dict': model.target_model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'config': self.config_manager.config,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'state_dim': model.state_dim,
                    'action_dim': model.action_dim,
                    'num_modes': model.num_modes,
                    'device': str(model.device),
                    'training_results': training_results or {},
                    'version': '1.0'
                }
            }
            
            # Guardar modelo
            torch.save(checkpoint, filepath)
            
            # Guardar metadatos adicionales en JSON
            metadata_file = filepath.replace('.pt', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(checkpoint['metadata'], f, indent=2)
            
            self.logger.info(f"✅ Modelo guardado en: {filepath}")
            self.logger.info(f"📄 Metadatos guardados en: {metadata_file}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando modelo: {e}")
            return None
    
    def load_model(self, model_path, model_instance=None):
        """
        Cargar modelo HNAF desde archivo
        
        Args:
            model_path: Ruta al archivo del modelo
            model_instance: Instancia de HNAFImproved (opcional)
            
        Returns:
            Tuple (model_instance, metadata)
        """
        try:
            # Cargar checkpoint con compatibilidad para PyTorch>=2.6
            # 1) Intentar con weights_only=True añadiendo globals seguros si es necesario
            checkpoint = None
            try:
                try:
                    # Allowlist para numpy scalar en safe unpickler
                    import numpy as np
                    from torch.serialization import add_safe_globals
                    add_safe_globals([np.core.multiarray.scalar])
                except Exception:
                    pass
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                # 2) Fallback explícito a weights_only=False (requiere confianza en el archivo)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Crear instancia del modelo si no se proporciona
            if model_instance is None:
                from hnaf_improved import HNAFImproved
                model_instance = HNAFImproved(checkpoint['config'], self.logger)
            
            # Cargar pesos
            model_instance.model.load_state_dict(checkpoint['model_state_dict'])
            model_instance.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            
            # Cargar optimizador si existe
            if 'optimizer_state_dict' in checkpoint:
                model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Poner en modo evaluación
            model_instance.model.eval()
            model_instance.target_model.eval()
            
            metadata = checkpoint.get('metadata', {})
            
            self.logger.info(f"✅ Modelo cargado desde: {model_path}")
            self.logger.info(f"📊 Metadatos: {metadata}")
            
            return model_instance, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo: {e}")
            return None, None
    
    def list_models(self):
        """Listar todos los modelos disponibles"""
        models = []
        
        for file in os.listdir(self.save_dir):
            if file.endswith('.pt'):
                filepath = os.path.join(self.save_dir, file)
                metadata_file = filepath.replace('.pt', '_metadata.json')
                
                metadata = {}
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                models.append({
                    'filename': file,
                    'filepath': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)),
                    'metadata': metadata
                })
        
        # Ordenar por fecha de modificación (más reciente primero)
        models.sort(key=lambda x: x['modified'], reverse=True)
        
        return models
    
    def get_best_model(self, metric='precision', higher_is_better=True):
        """
        Obtener el mejor modelo según una métrica específica
        
        Args:
            metric: Métrica a considerar ('precision', 'reward', etc.)
            higher_is_better: Si True, mayor valor es mejor
            
        Returns:
            Ruta al mejor modelo o None
        """
        models = self.list_models()
        
        if not models:
            return None
        
        # Filtrar modelos con métricas disponibles
        valid_models = []
        for model in models:
            training_results = model['metadata'].get('training_results', {})
            if metric in training_results:
                valid_models.append((model, training_results[metric]))
        
        if not valid_models:
            # Si no hay métricas, devolver el más reciente
            return models[0]['filepath']
        
        # Ordenar por métrica
        if higher_is_better:
            valid_models.sort(key=lambda x: x[1], reverse=True)
        else:
            valid_models.sort(key=lambda x: x[1])
        
        return valid_models[0][0]['filepath']
    
    def delete_old_models(self, keep_count=5):
        """
        Eliminar modelos antiguos, manteniendo solo los más recientes
        
        Args:
            keep_count: Número de modelos a mantener
        """
        models = self.list_models()
        
        if len(models) <= keep_count:
            self.logger.info(f"📁 Solo hay {len(models)} modelos, no se eliminan")
            return
        
        # Eliminar modelos antiguos
        for model in models[keep_count:]:
            try:
                os.remove(model['filepath'])
                
                # Eliminar archivo de metadatos si existe
                metadata_file = model['filepath'].replace('.pt', '_metadata.json')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                
                self.logger.info(f"🗑️ Eliminado modelo antiguo: {model['filename']}")
                
            except Exception as e:
                self.logger.error(f"❌ Error eliminando modelo {model['filename']}: {e}")
        
        self.logger.info(f"✅ Limpieza completada. Mantenidos {keep_count} modelos")

def get_model_saver():
    """Función helper para obtener instancia de ModelSaver"""
    return ModelSaver()

def save_training_checkpoint(model, training_results, filename=None):
    """
    Función helper para guardar checkpoint durante entrenamiento
    
    Args:
        model: Instancia de HNAFImproved
        training_results: Resultados del entrenamiento
        filename: Nombre personalizado del archivo
        
    Returns:
        Ruta del archivo guardado o None
    """
    saver = get_model_saver()
    return saver.save_model(model, training_results, filename)

def load_latest_model():
    """
    Función helper para cargar el modelo más reciente
    
    Returns:
        Tuple (model_instance, metadata) o (None, None)
    """
    saver = get_model_saver()
    models = saver.list_models()
    
    if not models:
        return None, None
    
    latest_model_path = models[0]['filepath']
    return saver.load_model(latest_model_path)

if __name__ == "__main__":
    # Script de prueba
    saver = get_model_saver()
    
    print("📁 Modelos disponibles:")
    models = saver.list_models()
    
    for i, model in enumerate(models):
        print(f"  {i+1}. {model['filename']}")
        print(f"     Tamaño: {model['size'] / 1024:.1f} KB")
        print(f"     Fecha: {model['modified']}")
        if model['metadata']:
            print(f"     Métricas: {model['metadata'].get('training_results', {})}")
        print()
    
    if models:
        print(f"🎯 Mejor modelo por precisión: {saver.get_best_model('precision')}") 