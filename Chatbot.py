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

# 🏪 Información de la Tienda
class InformacionTienda:
    """
    Autor: Dylan
    Información completa de MijoStore
    """

    def __init__(self):
        self.datos_tienda = {
            "nombre": "MijoStore",
            "email": "mijostore.online@gmail.com",
            "direccion_principal": "Calle Zela Nro 267, Tacna",
            "direccion_secundaria": "Cnel. Inclan 382-196, Tacna 23001",
            "telefonos": ["052632704", "+51952909892"],
            "whatsapp": "https://api.whatsapp.com/send/?phone=51952909892&text&type=phone_number&app_absent=0",
            "redes_sociales": {
                "facebook": "https://www.facebook.com/mijostore.tacna",
                "instagram": "https://www.instagram.com/mijostoretacna/?hl=es-la"
            },
            "sitio_web": "https://mijostore.pe/?fbclid=IwY2xjawLSUO1leHRuA2FlbQIxMABicmlkETF4RUxXN1NDZUt4TjhWbjE4AR6NUsghA-ogV12NB_MhC0Z0FGg_nwGllHl2Lx2PLTvDSgcN41cjJhwmhmS8GA_aem_drTG7tlGwTf7Ze24epUqDg",
            "google_maps": "https://maps.app.goo.gl/uc2KXhQr7bEHiZA86",
            "horarios": "Lunes a Sábado: 9:00 AM - 8:00 PM, Domingos: 10:00 AM - 6:00 PM",
            "especialidad": "Venta de celulares, accesorios y servicios técnicos"
        }

    def obtener_informacion_completa(self) -> str:
        """Obtiene toda la información de la tienda formateada"""
        info = self.datos_tienda
        return f"""
📱 **MIJOSTORE - TU TIENDA DE CELULARES EN TACNA**

📧 **Correo:** {info['email']}

📍 **Ubicaciones:**
• Principal: {info['direccion_principal']}
• Sucursal: {info['direccion_secundaria']}

📞 **Contáctanos:**
• Teléfono: {info['telefonos'][0]}
• Celular: {info['telefonos'][1]}
• WhatsApp: {info['whatsapp']}

🌐 **Redes Sociales:**
• Facebook: {info['redes_sociales']['facebook']}
• Instagram: {info['redes_sociales']['instagram']}

🗺️ **Ubicación en Google Maps:**
{info['google_maps']}

🌐 **Sitio Web:**
{info['sitio_web']}

🕒 **Horarios:** {info['horarios']}
🛍️ **Especialidad:** {info['especialidad']}
"""

    def obtener_contacto(self) -> str:
        """Obtiene información de contacto"""
        info = self.datos_tienda
        return f"""
📞 **CONTACTO MIJOSTORE**
• Teléfono fijo: {info['telefonos'][0]}
• Celular/WhatsApp: {info['telefonos'][1]}
• Email: {info['email']}
• WhatsApp directo: {info['whatsapp']}
"""

    def obtener_ubicacion(self) -> str:
        """Obtiene información de ubicación"""
        info = self.datos_tienda
        return f"""
📍 **UBICACIÓN MIJOSTORE**

🏪 **Tienda Principal:**
{info['direccion_principal']}

🏪 **Sucursal:**
{info['direccion_secundaria']}

🗺️ **Ver en Google Maps:**
{info['google_maps']}

🕒 **Horarios de atención:**
{info['horarios']}
"""

    def obtener_redes_sociales(self) -> str:
        """Obtiene información de redes sociales"""
        info = self.datos_tienda
        return f"""
🌐 **SÍGUENOS EN REDES SOCIALES**

📘 **Facebook:** {info['redes_sociales']['facebook']}
📸 **Instagram:** {info['redes_sociales']['instagram']}
🌍 **Sitio Web:** {info['sitio_web']}

¡Síguenos para ofertas exclusivas y novedades! 📱✨
"""
