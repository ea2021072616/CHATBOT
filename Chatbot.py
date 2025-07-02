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



# 🗄️ Base de Datos de Celulares
class BaseDatosCelulares:
    """
    Autor: Fabiola
    Simulación de base de datos con celulares modernos
    """

    def __init__(self):
        self.celulares = [
            Celular(
                id=1, marca="Samsung", modelo="Galaxy S24 Ultra", precio=4299.0,
                almacenamiento="256GB", ram="12GB",
                camara_principal="200MP", camara_frontal="12MP",
                pantalla="6.8\" Dynamic AMOLED", bateria="5000mAh",
                puntuacion_foto=10, puntuacion_rendimiento=10,
                caracteristicas_especiales=["S Pen", "Zoom 100x", "IA avanzada"]
            ),
            Celular(
                id=2, marca="iPhone", modelo="15 Pro Max", precio=5199.0,
                almacenamiento="256GB", ram="8GB",
                camara_principal="48MP", camara_frontal="12MP",
                pantalla="6.7\" Super Retina XDR", bateria="4441mAh",
                puntuacion_foto=9, puntuacion_rendimiento=10,
                caracteristicas_especiales=["Titanio", "Action Button", "USB-C"]
            ),
            Celular(
                id=3, marca="Xiaomi", modelo="14 Ultra", precio=3899.0,
                almacenamiento="512GB", ram="16GB",
                camara_principal="50MP Leica", camara_frontal="32MP",
                pantalla="6.73\" WQHD+", bateria="5300mAh",
                puntuacion_foto=9, puntuacion_rendimiento=9,
                caracteristicas_especiales=["Leica optics", "Carga 90W", "IP68"]
            ),
            Celular(
                id=4, marca="Google", modelo="Pixel 8 Pro", precio=3299.0,
                almacenamiento="256GB", ram="12GB",
                camara_principal="50MP", camara_frontal="10.5MP",
                pantalla="6.7\" LTPO OLED", bateria="5050mAh",
                puntuacion_foto=9, puntuacion_rendimiento=8,
                caracteristicas_especiales=["IA Tensor G3", "Magic Eraser", "Android puro"]
            ),
            Celular(
                id=5, marca="Samsung", modelo="Galaxy A55", precio=1799.0,
                almacenamiento="128GB", ram="8GB",
                camara_principal="50MP", camara_frontal="32MP",
                pantalla="6.6\" Super AMOLED", bateria="5000mAh",
                puntuacion_foto=7, puntuacion_rendimiento=7,
                caracteristicas_especiales=["Resistente al agua", "Carga rápida", "One UI"]
            ),
            Celular(
                id=6, marca="OnePlus", modelo="12", precio=2899.0,
                almacenamiento="256GB", ram="12GB",
                camara_principal="50MP Hasselblad", camara_frontal="32MP",
                pantalla="6.82\" LTPO AMOLED", bateria="5400mAh",
                puntuacion_foto=8, puntuacion_rendimiento=9,
                caracteristicas_especiales=["Hasselblad", "Carga 100W", "OxygenOS"]
            ),
            Celular(
                id=7, marca="Xiaomi", modelo="Redmi Note 13 Pro", precio=1299.0,
                almacenamiento="256GB", ram="8GB",
                camara_principal="200MP", camara_frontal="16MP",
                pantalla="6.67\" AMOLED", bateria="5100mAh",
                puntuacion_foto=8, puntuacion_rendimiento=6,
                caracteristicas_especiales=["200MP camera", "120Hz", "IP54"]
            ),
            Celular(
                id=8, marca="Realme", modelo="GT 5 Pro", precio=2199.0,
                almacenamiento="256GB", ram="12GB",
                camara_principal="50MP", camara_frontal="32MP",
                pantalla="6.78\" LTPO AMOLED", bateria="5400mAh",
                puntuacion_foto=7, puntuacion_rendimiento=8,
                caracteristicas_especiales=["Snapdragon 8 Gen 3", "Carga 100W", "Periscope zoom"]
            )
        ]

    def buscar_por_presupuesto(self, presupuesto_max: float) -> List[Celular]:
        """Filtra celulares por presupuesto máximo"""
        return [c for c in self.celulares if c.precio <= presupuesto_max]

    def buscar_por_marca(self, marca: str) -> List[Celular]:
        """Filtra celulares por marca"""
        return [c for c in self.celulares if marca.lower() in c.marca.lower()]

    def obtener_mejores_camara(self) -> List[Celular]:
        """Obtiene celulares con mejor puntuación de cámara"""
        return sorted(self.celulares, key=lambda x: x.puntuacion_foto, reverse=True)

    def obtener_todos(self) -> List[Celular]:
        """Obtiene todos los celulares"""
        return self.celulares