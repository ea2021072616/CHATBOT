# Instalación de dependencias (ejecutar en Google Colab)
!pip install --no-cache-dir llama-cpp-python==0.2.77 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
!pip install fastapi uvicorn nest-asyncio
!pip install langchain langchain_community langchain_core openai langchain-openai
!pip install gradio requests pydantic
!pip install supabase psycopg2-binary sqlparse pandas

!pip install llama-cpp-python
!pip install langchain langchain-core
!pip install fastapi uvicorn
!pip install gradio
!pip install pydantic
!pip install nest-asyncio

# 🔧 Configuración del Sistema
@dataclass
class ConfiguracionSistema:
    """
    Autor: Dylan
    Clase de configuración centralizada del sistema
    """
    modelo_path: str = "Meta-Llama-3-8B-Instruct-v2.Q3_K_M.gguf"
    puerto_servidor: int = 8000
    temperatura_llm: float = 0.7
    contexto_maximo: int = 2048
# 📱 Modelos de Datos con Pydantic (Salida Estructurada)
class Celular(BaseModel):
    """
    Autor: Fabiola
    Modelo de datos estructurados para celulares usando Pydantic
    """
    id: int = Field(description="ID único del celular")
    marca: str = Field(description="Marca del celular")
    modelo: str = Field(description="Modelo específico")
    precio: float = Field(description="Precio en soles")
    almacenamiento: str = Field(description="Capacidad de almacenamiento")
    ram: str = Field(description="Memoria RAM")
    camara_principal: str = Field(description="Resolución cámara principal")
    camara_frontal: str = Field(description="Resolución cámara frontal")
    pantalla: str = Field(description="Tamaño y tipo de pantalla")
    bateria: str = Field(description="Capacidad de batería")
    puntuacion_foto: int = Field(description="Puntuación calidad de fotos (1-10)")
    puntuacion_rendimiento: int = Field(description="Puntuación rendimiento (1-10)")
    caracteristicas_especiales: List[str] = Field(description="Características destacadas")

class RecomendacionEstructurada(BaseModel):
    """
    Autor: Erick
    Modelo de salida estructurada para recomendaciones
    """
    celular_recomendado: Celular = Field(description="Celular principal recomendado")
    alternativas: List[Celular] = Field(description="Lista de alternativas")
    razonamiento: str = Field(description="Explicación detallada de la recomendación")
    coincidencia_presupuesto: bool = Field(description="Si cumple con el presupuesto")
    puntuacion_match: float = Field(description="Puntuación de coincidencia (0-1)")

class ConsultaUsuario(BaseModel):
    """
    Autor: Dylan
    Modelo para estructurar las consultas de usuario
    """
    presupuesto_max: Optional[float] = Field(description="Presupuesto máximo en soles")
    prioridad_camara: bool = Field(description="Si prioriza calidad de cámara")
    prioridad_rendimiento: bool = Field(description="Si prioriza rendimiento")
    uso_principal: str = Field(description="Uso principal del dispositivo")
    marca_preferida: Optional[str] = Field(description="Marca preferida si tiene")
