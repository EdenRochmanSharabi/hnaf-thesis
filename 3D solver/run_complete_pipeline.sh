#!/bin/bash

# Script para ejecutar pipeline completo HNAF
# 1. Entrenamiento con monitoreo
# 2. Evaluación y generación de resultados para tesis

set -e  # Salir si hay error

echo "🚀 INICIANDO PIPELINE COMPLETO HNAF"
echo "======================================"
echo "Fecha: $(date)"
echo "Directorio: $(pwd)"
echo ""

# Función para mostrar progreso
show_progress() {
    echo "⏳ $1"
    echo "   $(date)"
    echo ""
}

# Función para verificar si el entrenamiento terminó
check_training_complete() {
    # Buscar en logs si el entrenamiento terminó
    if grep -q "Entrenamiento completado" logs/hnaf_system.log 2>/dev/null; then
        return 0
    fi
    return 1
}

# Función para esperar entrenamiento
wait_for_training() {
    echo "🔄 Esperando que termine el entrenamiento..."
    echo "   Monitoreando logs..."
    
    while ! check_training_complete; do
        echo "   ⏳ Entrenamiento en progreso... ($(date))"
        sleep 30  # Verificar cada 30 segundos
    done
    
    echo "✅ Entrenamiento completado!"
    echo ""
}

# Paso 1: Entrenamiento
show_progress "PASO 1: Iniciando entrenamiento con monitoreo"
echo "Ejecutando: python training_monitor.py --run 1"
echo ""

# Ejecutar entrenamiento en background
python training_monitor.py --run 1 &
TRAINING_PID=$!

echo "📊 Entrenamiento iniciado con PID: $TRAINING_PID"
echo "   Puedes monitorear el progreso en otra terminal con:"
echo "   tail -f logs/hnaf_system.log"
echo ""

# Esperar a que termine el entrenamiento
wait_for_training

# Verificar que el proceso de entrenamiento terminó correctamente
if kill -0 $TRAINING_PID 2>/dev/null; then
    echo "⚠️  Proceso de entrenamiento aún activo, esperando..."
    wait $TRAINING_PID
fi

echo "✅ Entrenamiento terminado exitosamente"
echo ""

# Pequeña pausa para asegurar que todo se guardó
sleep 5

# Paso 2: Evaluación y generación de resultados
show_progress "PASO 2: Iniciando evaluación y generación de resultados"
echo "Ejecutando: python generate_thesis_results.py --auto"
echo ""

# Ejecutar evaluación
python generate_thesis_results.py --auto

# Verificar resultado
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 PIPELINE COMPLETADO EXITOSAMENTE!"
    echo "======================================"
    echo "✅ Entrenamiento: Completado"
    echo "✅ Evaluación: Completada"
    echo "✅ Informes: Generados"
    echo ""
    echo "📁 Resultados disponibles en:"
    echo "   - tesis_resultados_*/"
    echo "   - modelos guardados en: models/"
    echo ""
    echo "📄 Archivos generados:"
    echo "   - informe_estabilidad.md"
    echo "   - informe_estabilidad.tex"
    echo "   - RESUMEN_TESIS.md"
    echo "   - Gráficas PNG"
    echo ""
    echo "🎓 ¡Listo para tu tesis!"
else
    echo ""
    echo "❌ ERROR: Falló la evaluación"
    echo "Revisa los logs para más detalles"
    exit 1
fi

echo "🏁 Pipeline completado en: $(date)" 