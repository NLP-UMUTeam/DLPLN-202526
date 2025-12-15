# DLPLN-2025-26
# Deep Learning para el Procesamiento del Lenguaje Natural

Este repositorio contiene material docente y práctico para la asignatura **Deep Learning para el Procesamiento del Lenguaje Natural (DLPLN)** del curso **2025–26**.

El repositorio está dividido en dos grandes bloques:

1) **Web Crawlers**  
2) **Ejemplos prácticos de uso del servidor ATLASv2 (HPC)**


### Web Crawlers
Esta carpeta contiene scripts sencillos de ejemplo para **recuperar datos de distintas fuentes online** y poder **compilar un corpus propio**.

El objetivo de estos scripts es servir como punto de partida para utilizar dichos datos en las prácticas de las **Partes II y III** de la asignatura.



### ATLASv2
Esta carpeta contiene recursos de **apoyo docente y técnico** para aprender a trabajar con un el servidor de computación **ATLASv2**.

#### ¿Qué es ATLASv2?
ATLASv2 es un **clúster** que permite ejecutar:
- entrenamientos de Deep Learning,
- experimentos con GPU,
- procesos costosos en tiempo o memoria,

que no serían viables en un ordenador personal.

En lugar de ejecutar programas directamente, los usuarios **envían trabajos (jobs)** al sistema, indicando:
- qué recursos necesitan (GPU, memoria, tiempo),
- qué programa quieren ejecutar.

#### ¿Qué es Slurm y por qué se usa?
**Slurm** es un **gestor de colas y recursos** que se encarga de:
- asignar CPUs, GPUs y memoria,
- planificar la ejecución de los trabajos,
- evitar que varios usuarios interfieran entre sí.


#### Idea general de los scripts en ATLASv2
Esta carpeta tiene recursos de **material de apoyo docente y técnico** para aprender a:
- ejecutar trabajos con GPU usando Slurm,
- trabajar con las carpetas temporales `/scratch`, y con el HOME.
- entrenar y usar modelos de Hugging Face para clasificación de textos, clasificación de secuencias, preguntas y respuestas
- aplicar distintos paradigmas de aprendizaje de In-Context Learning (zero-shot, few-shot, chain-of-thought) de Google Gemma. 
- Hacer SFT con LoRA de un modelo LlaMa (META)

#### Estructura general del repositorio
Dentro de la carpeta `atlasv2/` encontrarás scripts y utilidades organizados por funcionalidad:

- **Scripts de lanzamiento**
  - `bootstrap.sh`  
    Script base para lanzar trabajos en ATLASv2 usando Slurm y Apptainer.
    
    ```
    Ejemplo: sbatch bootstrap.sh zero_shot.py
    ```

- **Ficheros de utilidades**
  - `atlas_utils.py`  
    Funciones comunes para ATLASv2. Estas funcionales se utilizan a lo largo de todos los scripts
    - creación de carpetas en `/scratch`
    - configuración de cachés de Hugging Face
    - fijar semillas
    - detección de GPU / CPU

  - `prompting_utils.py`  
    Utilidades para trabajar con LLMs:
    - construcción de prompts tipo chat
    - normalización de etiquetas en clasificación binaria
    - extracción del texto generado por el modelo

  - `sft_utils.py`  
    Utilidades específicas para *Supervised Fine-Tuning (SFT)*:
    - formateo de ejemplos estilo Alpaca
    - estimación del ratio caracteres/tokens
    - preparación de datasets para TRL

- **Ejemplos de PLN / Deep Learning**
  - `sentence_classification.py`  
    Clasificación de textos (sentiment analysis).

  - `token_classification.py`  
    Reconocimiento de entidades nombradas (NER).

  - `question_answering.py`  
    Question Answering extractivo.

  - `zero_shot.py`  
    Clasificación zero-shot con LLMs.
    
  - `few_shot.py.py`  
    Clasificación few-shot con LLMs.

    - `chain_of_thought.py`  
    Clasificación few-shot con LLMs utilizando 

  - `sft.py`  
    Fine-tuning supervisado con LoRA (instruction tuning).

---

#### Uso de Hugging Face y `HF_TOKEN`

Algunos modelos (**Llama** y **Gema**) están protegidos (*gated models*) y requieren autenticación.

#### Cómo obtener un token de Hugging Face
1. Crear una cuenta o iniciar sesión en Hugging Face:
   - https://huggingface.co
2. Ir a:
   - https://huggingface.co/settings/tokens
3. Crear un token (con permisos de lectura es suficiente).

Pon el token en el fichero ```bootstrap.sh``` en la variable ```HF_TOKEN```
