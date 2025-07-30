# Contributing to HNAF Thesis

## 📋 Sobre este Proyecto

Este es un proyecto de **tesis de grado** sobre Hybrid Normalized Advantage Functions (HNAF). El código implementa un algoritmo de aprendizaje por refuerzo híbrido para sistemas de control con modos discretos y acciones continuas.

## 🎯 Tipos de Contribuciones Aceptadas

### ✅ Bienvenidas
- **Reportes de bugs**: Si encuentras algún error en el código
- **Sugerencias de mejora**: Ideas para optimizar el algoritmo
- **Documentación**: Mejoras en la documentación existente
- **Tests**: Casos de prueba adicionales
- **Optimizaciones**: Mejoras en rendimiento o estabilidad

### ❌ No Aceptadas
- Cambios que alteren significativamente la metodología de la tesis
- Modificaciones que no estén respaldadas por la literatura científica
- Cambios que comprometan la reproducibilidad de los resultados

## 🚀 Cómo Contribuir

### 1. Fork del Repositorio
1. Ve a [https://github.com/EdenRochmanSharabi/hnaf-thesis](https://github.com/EdenRochmanSharabi/hnaf-thesis)
2. Haz click en "Fork" en la esquina superior derecha
3. Clona tu fork localmente

### 2. Crear una Rama
```bash
git checkout -b feature/nombre-de-tu-mejora
```

### 3. Hacer Cambios
- Modifica solo los archivos necesarios
- Mantén el estilo de código consistente
- Añade comentarios explicativos cuando sea necesario
- Actualiza la documentación si es relevante

### 4. Commit y Push
```bash
git add .
git commit -m "feat: descripción clara de los cambios"
git push origin feature/nombre-de-tu-mejora
```

### 5. Crear Pull Request
1. Ve a tu fork en GitHub
2. Click en "Compare & pull request"
3. Describe claramente los cambios realizados
4. Menciona si es un bug fix, feature, o mejora de documentación

## 📝 Estilo de Código

### Python
- Usar **snake_case** para variables y funciones
- Usar **PascalCase** para clases
- Máximo 79 caracteres por línea
- Docstrings en formato Google
- Type hints cuando sea posible

### Ejemplo
```python
def calculate_reward(state: np.ndarray, action: np.ndarray) -> float:
    """
    Calcula la recompensa para un estado y acción dados.
    
    Args:
        state: Vector de estado del sistema
        action: Vector de acción aplicada
        
    Returns:
        float: Valor de la recompensa
    """
    return -np.linalg.norm(state - action)
```

## 🧪 Testing

### Ejecutar Tests Existentes
```bash
python test_hnaf_improvements.py
```

### Añadir Nuevos Tests
- Crear archivos con prefijo `test_`
- Usar nombres descriptivos para las funciones de test
- Incluir casos edge y casos de error

## 📚 Documentación

### Estructura de Documentación
- `README.md`: Documentación principal del proyecto
- `RESUMEN_MEJORAS_HNAF.md`: Detalles técnicos de mejoras
- `CONTRIBUTING.md`: Esta guía
- Docstrings en el código: Documentación de funciones y clases

### Actualizar Documentación
- Mantener la documentación actualizada con los cambios
- Usar ejemplos claros y concisos
- Incluir diagramas cuando sea útil

## 🐛 Reportar Bugs

### Información Requerida
1. **Descripción del bug**: Qué esperabas vs qué pasó
2. **Pasos para reproducir**: Instrucciones detalladas
3. **Entorno**: Versión de Python, dependencias, sistema operativo
4. **Logs**: Mensajes de error completos
5. **Screenshots**: Si es relevante

### Template de Bug Report
```markdown
## Bug Report

### Descripción
[Describe el bug claramente]

### Pasos para Reproducir
1. [Paso 1]
2. [Paso 2]
3. [Paso 3]

### Comportamiento Esperado
[Describe qué debería pasar]

### Comportamiento Actual
[Describe qué está pasando]

### Entorno
- Python: [versión]
- PyTorch: [versión]
- Sistema Operativo: [OS]
- Otros: [dependencias relevantes]

### Logs de Error
```
[Pega aquí los logs completos]
```
```

## 💡 Sugerencias de Mejora

### Criterios de Evaluación
- **Relevancia**: ¿Es útil para la investigación?
- **Implementabilidad**: ¿Es técnicamente factible?
- **Impacto**: ¿Mejora significativamente el algoritmo?
- **Documentación**: ¿Está bien respaldada por la literatura?

### Template de Feature Request
```markdown
## Feature Request

### Descripción
[Describe la mejora propuesta]

### Motivación
[¿Por qué es necesaria esta mejora?]

### Propuesta de Implementación
[Describe cómo se podría implementar]

### Referencias
[Papers, artículos o recursos relevantes]

### Impacto Esperado
[¿Qué beneficios traería?]
```

## 📞 Contacto

Para consultas específicas sobre la tesis o el algoritmo:

- **Autor**: Eden Rochman
- **Email**: eden@example.com
- **GitHub**: [@EdenRochmanSharabi](https://github.com/EdenRochmanSharabi)

## 📄 Licencia

Al contribuir a este proyecto, aceptas que tus contribuciones serán licenciadas bajo la [MIT License](LICENSE).

---

**Gracias por contribuir a este proyecto de investigación académica!** 🎓 