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
    
    # 🧠 Clase Base para Modelos de IA
class ModeloAIBase(ABC):
    """
    Autor: Erick
    Clase abstracta base para modelos de inteligencia artificial
    """

    @abstractmethod
    def inicializar_modelo(self) -> None:
        """Inicializa el modelo de IA"""
        pass

    @abstractmethod
    def generar_respuesta(self, mensajes: List[Dict]) -> str:
        """Genera una respuesta basada en los mensajes"""
        pass
# 🚀 Servidor LLM con FastAPI
class ServidorLLM(ModeloAIBase):
    """
    Autor: Erick
    Servidor que expone el modelo Llama como API compatible con OpenAI
    """

    def __init__(self, config: ConfiguracionSistema):
        self.config = config
        self.modelo = None
        self.app = FastAPI(title="EmpresaBot LLM Server")
        self.inicializar_modelo()
        self._configurar_rutas()

    def inicializar_modelo(self) -> None:
        """
        Autor: Erick
        Inicializa el modelo Llama con configuraciones optimizadas
        """
        print("🤖 Inicializando modelo Llama 3...")
        self.modelo = Llama(
            model_path=self.config.modelo_path,
            n_ctx=self.config.contexto_maximo,
            n_gpu_layers=-1,  # Usar GPU si está disponible
            chat_format="chatml",
            verbose=False
        )
        print("✅ Modelo inicializado correctamente")

    def _configurar_rutas(self) -> None:
        """
        Autor: Erick
        Configura las rutas de la API
        """
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._procesar_completacion_chat(request)

    async def _procesar_completacion_chat(self, request: Request) -> JSONResponse:
        """
        Autor: Erick
        Procesa las solicitudes de completación de chat
        """
        body = await request.json()
        mensajes = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", self.config.temperatura_llm)

        if not stream:
            return self._generar_respuesta_sincrona(mensajes, temperature)
        else:
            return self._generar_respuesta_stream(mensajes, temperature)

    def _generar_respuesta_sincrona(self, mensajes: List[Dict], temperature: float) -> JSONResponse:
        """
        Autor: Erick
        Genera respuesta síncrona
        """
        res = self.modelo.create_chat_completion(messages=mensajes, temperature=temperature)
        return JSONResponse(content={
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-3-8b-instruct",
            "choices": [{
                "index": 0,
                "message": res["choices"][0]["message"],
                "finish_reason": res["choices"][0].get("finish_reason", "stop")
            }],
            "usage": res.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        })

    def _generar_respuesta_stream(self, mensajes: List[Dict], temperature: float) -> StreamingResponse:
        """
        Autor: Erick
        Genera respuesta en modo streaming
        """
        def stream_generator():
            for chunk in self.modelo.create_chat_completion(
                messages=mensajes, temperature=temperature, stream=True
            ):
                choice = chunk["choices"][0]
                delta = choice.get("delta", {}) or choice.get("message", {})
                yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4()}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'llama-3-8b-instruct', 'choices': [{'delta': delta, 'index': 0, 'finish_reason': choice.get('finish_reason')}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    def generar_respuesta(self, mensajes: List[Dict]) -> str:
        """
        Autor: Erick
        Implementación del método abstracto
        """
        res = self.modelo.create_chat_completion(messages=mensajes)
        return res["choices"][0]["message"]["content"]

    def iniciar_servidor(self) -> None:
        """
        Autor: Erick
        Inicia el servidor FastAPI en un hilo separado
        """
        def ejecutar_servidor():
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=self.config.puerto_servidor, log_level="error")

        nest_asyncio.apply()
        hilo_servidor = threading.Thread(target=ejecutar_servidor, daemon=True)
        hilo_servidor.start()
        print(f"🌐 Servidor LLM iniciado en puerto {self.config.puerto_servidor}")

# 💬 Gestor de Conversaciones
class GestorConversacion:
    """
    Autor: Dylan
    Maneja el historial y contexto de las conversaciones
    """

    def __init__(self):
        self.conversaciones: Dict[str, List[tuple]] = {}
        self.metadatos: Dict[str, Dict] = {}

    def crear_sesion(self, id_sesion: str) -> None:
        """
        Autor: Dylan
        Crea una nueva sesión de conversación
        """
        self.conversaciones[id_sesion] = []
        self.metadatos[id_sesion] = {
            "fecha_inicio": datetime.now(),
            "total_mensajes": 0,
            "estado": "activa"
        }

    def agregar_mensaje(self, id_sesion: str, rol: str, mensaje: str) -> None:
        """
        Autor: Dylan
        Agrega un mensaje al historial de conversación
        """
        if id_sesion not in self.conversaciones:
            self.crear_sesion(id_sesion)

        self.conversaciones[id_sesion].append((rol, mensaje))
        self.metadatos[id_sesion]["total_mensajes"] += 1

    def obtener_historial(self, id_sesion: str) -> List[tuple]:
        """
        Autor: Dylan
        Obtiene el historial completo de una conversación
        """
        return self.conversaciones.get(id_sesion, [])

    def obtener_contexto_formateado(self, id_sesion: str) -> str:
        """
        Autor: Dylan
        Obtiene el contexto formateado para el modelo
        """
        historial = self.obtener_historial(id_sesion)
        return "\n".join([f"{rol}: {mensaje}" for rol, mensaje in historial])
