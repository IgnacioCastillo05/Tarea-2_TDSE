# Tarea-2_TDSE
Trabajo de TDSE sobre regresión logística

## Paso 1

### Feature Selection

Se seleccionaron 8 características basándose en tres criterios principales:

**1. Relevancia Clínica:**
Las características elegidas son indicadores médicos establecidos para el diagnóstico 
de enfermedad cardíaca según la literatura médica:
- **Age**: Principal factor de riesgo no modificable
- **Cholesterol & BP**: Factores de riesgo cardiovascular modificables
- **Max HR**: Indicador de capacidad funcional cardíaca
- **ST depression**: Marcador ECG de isquemia miocárdica
- **Number of vessels fluro**: Resultado de angiografía coronaria
- **Chest pain type**: Síntoma clínico diferencial
- **Thallium**: Prueba de perfusión miocárdica

**2. Análisis Exploratorio (EDA):**
Durante el EDA se observó que todas las características seleccionadas presentan:
- Distribuciones con variabilidad adecuada (sin valores constantes)
- Rangos que cubren espectros clínicamente significativos
- Presencia de casos en zonas de riesgo (ej: Cholesterol >200 mg/dL, BP >140 mmHg)

Por ejemplo:
- **Age**: Concentración en 48-67 años (rango de mayor riesgo cardiovascular)
- **Cholesterol**: 59% de pacientes con valores >200 mg/dL (límite de riesgo)
- **ST depression**: Distribución sesgada con valores anormales (>0) en 36% de casos
- **Max HR**: Variabilidad de 71-202 bpm, capturando diferentes niveles de condición física

**3. Completitud de Datos:**
Las 8 características seleccionadas no presentan valores nulos (270/270 registros completos),
lo que elimina la necesidad de imputación y preserva la integridad del dataset.

Estas 8 features representan un balance entre:
- Diversidad de información (demográfica, síntomas, pruebas de laboratorio, ECG, imaging)
- Parsimonia del modelo (evitar overfitting con demasiadas features)
- Interpretabilidad clínica (todas son mediciones estándar en práctica médica)

## Paso 2


