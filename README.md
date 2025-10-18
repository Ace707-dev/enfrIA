# Clasificador de Frutas y Verduras: Saludables vs Podridas

Sistema de clasificación en tiempo real que utiliza YOLO para detectar si frutas y verduras están saludables o podridas a través de la cámara web.

## Descripción

Este proyecto implementa un clasificador de imágenes basado en deep learning que:
- Captura imágenes en tiempo real desde la cámara web
- Clasifica automáticamente frutas/verduras como *healthy* (saludables) o *rotten* (podridas)
- Organiza las imágenes clasificadas en carpetas
- Genera logs con las predicciones y nivel de confianza
- Soporta aceleración por GPU (CUDA)

## Características

- ✅ Clasificación en tiempo real con YOLO
- ✅ Captura automática cada 5 segundos
- ✅ Captura manual con tecla 'C'
- ✅ Feedback visual en pantalla
- ✅ Organización automática de imágenes
- ✅ Registro de predicciones en CSV
- ✅ Soporte GPU/CPU automático
- ✅ Descarga automática del modelo entrenado

## Requisitos

### Software
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Cámara web conectada

### Hardware (Recomendado)
- GPU NVIDIA con CUDA (opcional, para mejor rendimiento)
- Mínimo 4GB RAM
- Espacio en disco: ~2GB

## Instalación

### 1. Clonar el repositorio
powershell
git clone https://github.com/Ace707-dev/enfrIA.git
cd enfrIA


### 2. Crear entorno virtual (recomendado)
powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1


### 3. Instalar dependencias

*Para CPU:*
powershell
pip install -r requirements.txt


*Para GPU con CUDA:*
powershell
# Primero instalar PyTorch con soporte CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Luego el resto de dependencias
pip install -r requirements.txt


## Uso

### Ejecutar el clasificador en tiempo real
powershell
python main.py


### Controles
- **C**: Capturar y clasificar imagen manualmente
- **Q**: Salir del programa
- *Automático*: Captura cada 5 segundos

### Clasificar imágenes individuales o carpetas
powershell
# Clasificar una imagen
python classify.py .\runs\healthy-rotten10\weights\best.pt .\ruta\a\imagen.jpg

# Clasificar carpeta de imágenes
python classify.py .\runs\healthy-rotten10\weights\best.pt .\archive\Fruit And Vegetable Diseases Dataset_sorted\val\healthy

# Usar webcam
python classify.py .\runs\healthy-rotten10\weights\best.pt 0


### Variables de entorno (opcional)
powershell
$env:MODEL_PATH="runs/healthy-rotten10/weights/best.pt"
$env:DATA_DIR="archive/Fruit And Vegetable Diseases Dataset_sorted"
$env:MODEL_URL="https://github.com/Ace707-dev/enfrIA/releases/download/TrainedModel/best.pt"


## Estructura del Proyecto


Proyecto/
├── main.py                    # Script principal con cámara en tiempo real
├── train.py                   # Script de entrenamiento
├── classify.py                # Clasificación de imágenes individuales
├── entrenamiento.py           # Verificación de GPU
├── requirements.txt           # Dependencias
├── README.md                  # Este archivo
├── prediction_log.csv         # Registro de predicciones
├── capturas/                  # Imágenes capturadas
├── runs/
│   └── healthy-rotten10/      # Modelos entrenados
│       ├── args.yaml          # Configuración del entrenamiento
│       ├── results.csv        # Métricas de entrenamiento
│       └── weights/
│           ├── best.pt        # Mejor modelo
│           └── last.pt        # Último checkpoint
└── archive/
    └── Fruit And Vegetable Diseases Dataset_sorted/
        ├── healthy/           # Imágenes clasificadas como saludables
        ├── rotten/            # Imágenes clasificadas como podridas
        ├── train/             # Datos de entrenamiento
        │   ├── healthy/
        │   └── rotten/
        └── val/               # Datos de validación
            ├── healthy/
            └── rotten/


## Dataset

El proyecto usa una estructura estándar de YOLOv8 para clasificación:


archive/Fruit And Vegetable Diseases Dataset_sorted/
  train/
    healthy/
    rotten/
  val/
    healthy/
    rotten/


*Nota*: No se necesita archivo YAML para clasificación; las clases se derivan automáticamente de los nombres de carpetas.

Los resultados se guardarán en runs/healthy-rotten/.

### Parámetros de entrenamiento (train.py)
- *epochs*: Número de épocas de entrenamiento
- *imgsz*: Tamaño de imagen (224 recomendado)
- *batch*: Tamaño de batch (ajustar según memoria disponible)
- *device*: 'cuda' para GPU, 'cpu' para CPU



## Resultados

### Estadísticas finales
Al terminar una sesión, main.py muestra:

============================================================
✓ Se capturaron y clasificaron 15 imagen(es)
  Healthy: 8
  Rotten:  7
  Confianza promedio: 91.24%
============================================================


### Rendimiento
- *GPU (CUDA)*: ~30-60 FPS
- *CPU*: ~5-15 FPS
- *Precisión*: Variable según iluminación y calidad de imagen


*Proyecto académico* - 2do Parcial de Procesamiento de Imágenes - 2025