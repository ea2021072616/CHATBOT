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


# 🧠 Motor Principal del Chatbot
class ChatbotEngine:
    """
    Autor: Dylan
    Motor principal que coordina todas las funcionalidades del chatbot
    """

    def __init__(self, config: ConfiguracionSistema):
        self.config = config
        self.gestor_conversacion = GestorConversacion()
        self.base_datos = BaseDatosCelulares()
        self.info_tienda = InformacionTienda()
        self.llm_client = None
        self.cadena_chat: Optional[Runnable] = None
        self.cadena_recomendacion: Optional[Runnable] = None
        self.cadenas_paralelas: Optional[Runnable] = None
        self._inicializar_llm()
        self._configurar_cadenas()

    def _inicializar_llm(self) -> None:
        """
        Autor: Erick
        Inicializa el cliente LLM
        """
        self.llm_client = ChatOpenAI(
            base_url=f'http://localhost:{self.config.puerto_servidor}/v1',
            openai_api_key="not-needed",
            temperature=self.config.temperatura_llm
        )

    def _configurar_cadenas(self) -> None:
        """
        Autor: Dylan
        Configura las cadenas principales y paralelas del sistema
        """
        # Cadena principal de chat
        prompt_chat = ChatPromptTemplate.from_messages([
            ("system", """Eres Mijito, el asistente virtual amigable y experto de MijoStore en Tacna. Tu personalidad es:
- 🤗 Amable y cercano
- 🧠 Conocedor de tecnología
- 💼 Profesional pero accesible
- 🎯 Enfocado en ayudar al cliente

INFORMACIÓN DE LA TIENDA:
- 🏪 Nombre: MijoStore
- 📍 Ubicación: Calle Zela Nro 267, Tacna / Cnel. Inclan 382-196, Tacna 23001
- 📞 Contacto: 052632704, +51952909892
- 📧 Email: mijostore.online@gmail.com
- 🛍️ Especialidad: Venta de celulares, accesorios y servicios técnicos

INSTRUCCIONES:
1. 🤝 Saluda de manera amistosa si es el primer mensaje
2. 🧐 Analiza lo que realmente necesita el usuario
3. 💡 Ofrece soluciones específicas y útiles
4. 😊 Usa emojis para hacer la conversación más agradable
5. 🗣️ Habla de manera natural y conversacional
6. ❓ Haz preguntas para entender mejor sus necesidades
7. ✨ Siempre termina ofreciendo más ayuda

Para recomendaciones de celulares, pregunta sobre:
- 💰 Presupuesto máximo
- 📸 Si prioriza cámara
- ⚡ Si prioriza rendimiento
- 🎮 Uso principal (fotos, gaming, trabajo, etc.)
- 🏷️ Marca preferida

CONTEXTO: {contexto}
HISTORIAL: {chat_conversation}

Responde de manera natural, amigable y profesional. Usa emojis apropiados."""),
            ("placeholder", "{chat_conversation}")
        ])

        # Cadena de análisis de consulta (Runnable Function)
        def analizar_consulta(inputs: Dict) -> Dict:
            """Función ejecutable para analizar consultas del usuario"""
            consulta = inputs["mensaje"]

            # Análisis básico con regex
            presupuesto = None
            presupuesto_match = re.search(r'(\d+(?:\.\d+)?)\s*soles?', consulta.lower())
            if presupuesto_match:
                presupuesto = float(presupuesto_match.group(1))

            prioridad_camara = any(word in consulta.lower() for word in ['foto', 'camara', 'cámara', 'selfie'])
            prioridad_rendimiento = any(word in consulta.lower() for word in ['rápido', 'gaming', 'juego', 'rendimiento'])

            return {
                **inputs,
                "presupuesto_detectado": presupuesto,
                "prioridad_camara": prioridad_camara,
                "prioridad_rendimiento": prioridad_rendimiento
            }

        # Cadena de recomendación estructurada
        prompt_recomendacion = ChatPromptTemplate.from_messages([
            ("system", """Eres Mijito, el asistente virtual experto y amigable de MijoStore. Tu objetivo es dar recomendaciones personalizadas y fáciles de entender.

CRITERIOS USUARIO:
- Presupuesto: {presupuesto_detectado}
- Prioridad cámara: {prioridad_camara}
- Prioridad rendimiento: {prioridad_rendimiento}

BASE DE DATOS DISPONIBLE:
{celulares_disponibles}

INSTRUCCIONES PARA RESPUESTA AMIGABLE:
1. Saluda de manera amistosa y menciona que entiendes sus necesidades
2. Analiza las opciones de forma sencilla y conversacional
3. Recomienda 1 celular principal explicando POR QUÉ es perfecto para él
4. Menciona 2-3 alternativas brevemente
5. Usa emojis, formato atractivo y lenguaje cercano
6. Incluye datos específicos pero de forma amigable
7. Termina preguntando si necesita más información

FORMATO DE RESPUESTA:
🎯 **¡Perfecto! He encontrado el celular ideal para ti**

📱 **MI RECOMENDACIÓN PRINCIPAL:**
**[Marca] [Modelo]** - ¡Esta es mi elección!

💰 **Precio:** S/[precio]
📸 **Cámara:** [detalles de cámara]
🚀 **Rendimiento:** [detalles de rendimiento]
🔋 **Batería:** [capacidad]
💾 **Almacenamiento:** [capacidad]

🤔 **¿Por qué este celular?**
[Explicación personalizada y amigable]

🔄 **Otras opciones que podrían interesarte:**
• **[Alternativa 1]** - S/[precio] - [breve descripción]
• **[Alternativa 2]** - S/[precio] - [breve descripción]

✨ **En MijoStore tenemos todos estos modelos disponibles** ¿Te gustaría conocer más detalles de alguno o necesitas información sobre garantías y formas de pago?

Responde de manera natural, amigable y profesional."""),
            ("human", "{mensaje}")
        ])

        # Funciones ejecutables (Runnables)
        analizar_runnable = RunnableLambda(analizar_consulta)

        def obtener_celulares_contexto(inputs: Dict) -> Dict:
            """Runnable para obtener celulares relevantes"""
            presupuesto = inputs.get("presupuesto_detectado")

            if presupuesto:
                celulares = self.base_datos.buscar_por_presupuesto(presupuesto)
            else:
                celulares = self.base_datos.obtener_todos()

            celulares_texto = "\n".join([
                f"ID: {c.id}, {c.marca} {c.modelo}, S/{c.precio}, "
                f"Cámara: {c.camara_principal}, RAM: {c.ram}, "
                f"Puntuación foto: {c.puntuacion_foto}/10, "
                f"Puntuación rendimiento: {c.puntuacion_rendimiento}/10"
                for c in celulares[:6]  # Limitar para no saturar el contexto
            ])

            return {
                **inputs,
                "celulares_disponibles": celulares_texto
            }

        obtener_celulares_runnable = RunnableLambda(obtener_celulares_contexto)

        # Cadenas combinadas y paralelas
        self.cadena_chat = prompt_chat | self.llm_client | StrOutputParser()

        # Cadena de recomendación con pipeline
        self.cadena_recomendacion = (
            analizar_runnable |
            obtener_celulares_runnable |
            prompt_recomendacion |
            self.llm_client |
            StrOutputParser()
        )

        # Cadenas paralelas para análisis simultáneo
        self.cadenas_paralelas = RunnableParallel({
            "respuesta_general": self.cadena_chat,
            "analisis_consulta": analizar_runnable,
            "contexto_celulares": obtener_celulares_runnable
        })

    def _es_consulta_celular(self, mensaje: str) -> bool:
        """
        Autor: Fabiola
        Determina si la consulta es sobre recomendación de celulares
        """
        mensaje_lower = mensaje.lower()

        # Palabras clave que indican búsqueda de celulares
        palabras_recomendacion = [
            'recomienda', 'recomendación', 'necesito', 'quiero', 'busco',
            'ayuda', 'ayudame', 'ayúdame', 'gaming', 'juego', 'fotos',
            'camara', 'cámara', 'presupuesto', 'barato', 'económico'
        ]

        # Palabras que indican dispositivos
        palabras_dispositivo = ['celular', 'teléfono', 'telefono', 'smartphone', 'móvil', 'movil']

        # Debe tener al menos una palabra de recomendación Y una de dispositivo
        # O mencionar gaming/fotos + celular/smartphone
        tiene_recomendacion = any(palabra in mensaje_lower for palabra in palabras_recomendacion)
        tiene_dispositivo = any(palabra in mensaje_lower for palabra in palabras_dispositivo)

        # Casos específicos de gaming o fotos
        es_gaming_o_fotos = any(palabra in mensaje_lower for palabra in ['gaming', 'juego', 'fotos', 'camara', 'cámara'])

        return (tiene_recomendacion and tiene_dispositivo) or (es_gaming_o_fotos and tiene_dispositivo)


    def _es_consulta_tienda(self, mensaje: str) -> bool:
        """
        Autor: Dylan
        Determina si la consulta es sobre información de la tienda
        """
        mensaje_lower = mensaje.lower()

        # Si ya es una consulta de celular, no es de tienda
        if self._es_consulta_celular(mensaje):
            return False

        palabras_ubicacion = ['ubicación', 'ubicacion', 'dirección', 'direccion', 'donde', 'dónde', 'maps', 'mapa']
        palabras_contacto = ['contacto', 'teléfono', 'telefono', 'whatsapp', 'correo', 'email', 'llamar']
        palabras_redes = ['facebook', 'instagram', 'redes', 'sociales', 'pagina', 'página', 'web', 'sitio']
        palabras_horarios = ['horario', 'horarios', 'hora', 'horas', 'abierto', 'cerrado', 'atienden']
        palabras_tienda = ['tienda', 'store', 'mijo', 'mijostore', 'negocio', 'local']

        return (any(palabra in mensaje_lower for palabra in palabras_ubicacion) or
                any(palabra in mensaje_lower for palabra in palabras_contacto) or
                any(palabra in mensaje_lower for palabra in palabras_redes) or
                any(palabra in mensaje_lower for palabra in palabras_horarios) or
                any(palabra in mensaje_lower for palabra in palabras_tienda))

    def _procesar_consulta_tienda(self, mensaje: str) -> str:
        """
        Autor: Dylan
        Procesa consultas específicas sobre información de la tienda
        """
        mensaje_lower = mensaje.lower()

        # Detectar tipo específico de consulta
        if any(palabra in mensaje_lower for palabra in ['ubicación', 'ubicacion', 'dirección', 'direccion', 'donde', 'dónde', 'maps']):
            return self.info_tienda.obtener_ubicacion()

        elif any(palabra in mensaje_lower for palabra in ['contacto', 'teléfono', 'telefono', 'whatsapp', 'llamar', 'celular']):
            return self.info_tienda.obtener_contacto()

        elif any(palabra in mensaje_lower for palabra in ['facebook', 'instagram', 'redes', 'sociales', 'pagina', 'web']):
            return self.info_tienda.obtener_redes_sociales()

        else:
            # Información completa si no es específica
            return self.info_tienda.obtener_informacion_completa()
def procesar_mensaje(self, mensaje: str, id_sesion: str = "default") -> str:
        """
        Autor: Dylan
        Procesa un mensaje usando cadenas y análisis inteligente
        """
        # Agregar mensaje del usuario al historial
        self.gestor_conversacion.agregar_mensaje(id_sesion, "user", mensaje)

        # Obtener historial formateado
        historial = self.gestor_conversacion.obtener_historial(id_sesion)

        # Decidir qué tipo de respuesta dar basado en el contenido
        if self._es_consulta_celular(mensaje):
            # Usar cadena de recomendación para consultas de celulares
            inputs = {
                "mensaje": mensaje,
                "chat_conversation": historial
            }
            respuesta = self.cadena_recomendacion.invoke(inputs)
        elif self._es_consulta_tienda(mensaje):
            # Respuesta directa sobre información de la tienda
            respuesta = self._procesar_consulta_tienda(mensaje)
        else:
            # Usar cadena general para otras consultas
            inputs = {
                "contexto": "Eres un asistente de MijoStore, tienda especializada en celulares en Tacna",
                "chat_conversation": historial
            }
            respuesta = self.cadena_chat.invoke(inputs)

        # Agregar respuesta al historial
        self.gestor_conversacion.agregar_mensaje(id_sesion, "ai", respuesta)

        return respuesta

def procesar_recomendacion_estructurada(self, consulta: ConsultaUsuario) -> RecomendacionEstructurada:
        """
        Autor: Dylan
        Procesa una recomendación usando salida estructurada con Pydantic
        """
        # Filtrar celulares por criterios
        celulares_filtrados = self.base_datos.obtener_todos()

        if consulta.presupuesto_max:
            celulares_filtrados = [c for c in celulares_filtrados if c.precio <= consulta.presupuesto_max]

        if consulta.marca_preferida:
            celulares_filtrados = [c for c in celulares_filtrados if consulta.marca_preferida.lower() in c.marca.lower()]

        # Ordenar por criterios de prioridad
        if consulta.prioridad_camara:
            celulares_filtrados.sort(key=lambda x: x.puntuacion_foto, reverse=True)
        elif consulta.prioridad_rendimiento:
            celulares_filtrados.sort(key=lambda x: x.puntuacion_rendimiento, reverse=True)
        else:
            # Ordenar por mejor relación calidad-precio
            celulares_filtrados.sort(key=lambda x: (x.puntuacion_foto + x.puntuacion_rendimiento) / (x.precio / 1000), reverse=True)

        if not celulares_filtrados:
            # Devolver el más cercano al presupuesto si no hay matches exactos
            celulares_filtrados = sorted(self.base_datos.obtener_todos(), key=lambda x: abs(x.precio - (consulta.presupuesto_max or 2000)))

        # Crear recomendación estructurada
        principal = celulares_filtrados[0]
        alternativas = celulares_filtrados[1:4]  # Hasta 3 alternativas

        # Calcular puntuación de match
        puntuacion = 0.0
        if consulta.presupuesto_max and principal.precio <= consulta.presupuesto_max:
            puntuacion += 0.4
        if consulta.prioridad_camara and principal.puntuacion_foto >= 8:
            puntuacion += 0.3
        if consulta.prioridad_rendimiento and principal.puntuacion_rendimiento >= 8:
            puntuacion += 0.3

        return RecomendacionEstructurada(
            celular_recomendado=principal,
            alternativas=alternativas,
            razonamiento=f"Recomiendo el {principal.marca} {principal.modelo} porque cumple con tus criterios principales: "
                        f"{'excelente cámara' if consulta.prioridad_camara else ''} "
                        f"{'alto rendimiento' if consulta.prioridad_rendimiento else ''} "
                        f"y está {'dentro de tu presupuesto' if consulta.presupuesto_max and principal.precio <= consulta.presupuesto_max else 'cerca de tu rango de precio'}.",
            coincidencia_presupuesto=consulta.presupuesto_max is None or principal.precio <= consulta.presupuesto_max,
            puntuacion_match=min(puntuacion, 1.0)
        )

# 🎨 Interfaz de Usuario con Gradio
class InterfazWeb:
    """
    Autor: Fabiola
    Interfaz web moderna y responsive usando Gradio
    """

    def __init__(self, chatbot_engine: ChatbotEngine):
        self.chatbot_engine = chatbot_engine
        self.app = None
        self._crear_interfaz()
def _crear_interfaz(self) -> None:
        """
        Autor: Fabiola
        Crea la interfaz web con diseño moderno
        """
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Mijito Store - Asistente Virtual",
            css=self._obtener_estilos_css()
        ) as self.app:

            # Header con diseño corporativo
            gr.HTML("""
            <div style="text-align: center; background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);">
                <h1 style="color: white; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: bold;">
                    📱 Mijito Store
                </h1>
                <h2 style="color: #fff; font-size: 1.3em; margin: 10px 0 0 0; opacity: 0.95; font-weight: 300;">
                    Tu Asistente Virtual Inteligente
                </h2>
                <p style="color: #fff; margin: 15px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                    🤖 Powered by IA Avanzada • 🏪 Tacna, Perú • 📞 052632704
                </p>
            </div>
            """)

            # Estado del sistema con diseño mejorado
            with gr.Row():
                estado_sistema = gr.HTML("""
                <div style="background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.2);">
                    <span style="color: white; font-weight: bold; font-size: 1.1em;">
                        ✅ Sistema Activo - Llama 3 8B + LangChain
                    </span>
                </div>
                """)

            # Pestañas principales
            with gr.Tabs():
                # Pestaña 1: Chat libre
                with gr.TabItem("💬 Chat Libre"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            historial_chat = gr.Chatbot(
                                label="💬 Conversación",
                                height=400,
                                type='messages',
                                value=[{
                                    "role": "assistant",
                                    "content": """🎉 **¡Hola! Soy Mijito, tu asistente virtual de MijoStore** 📱

¡Bienvenido! Estoy aquí para ayudarte a encontrar el celular perfecto según tus necesidades y presupuesto. Con tecnología de IA avanzada, puedo darte recomendaciones personalizadas.

### 🤖 ¿En qué puedo ayudarte hoy?
- 📱 Recomendarte el celular ideal para ti
- 💰 Filtrar opciones por tu presupuesto
- 📸 Encontrar los mejores para fotografía
- 🎮 Celulares perfectos para gaming
- 📍 Información sobre nuestras tiendas
- 📞 Datos de contacto y ubicación

### 💡 **¡Haz clic en una pregunta rápida para empezar!**"""
                                }]
                            )

                            # Botones de preguntas rápidas con diseño corporativo
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid #ff6b35;">
                                <h3 style="color: #ff6b35; margin: 0 0 15px 0; font-weight: bold;">🚀 Preguntas Rápidas</h3>
                                <p style="color: #9a3412; margin: 0; font-size: 0.95em;">Haz clic en cualquier botón para obtener respuestas instantáneas</p>
                            </div>
                            """)

                            with gr.Row():
                                btn_gaming = gr.Button(
                                    "🎮 Gaming 3000 soles",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes="btn-gaming"
                                )
                                btn_fotos = gr.Button(
                                    "📸 Mejor cámara 2000 soles",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes="btn-fotos"
                                )

                            with gr.Row():
                                btn_economico = gr.Button(
                                    "💰 Celular económico 1500 soles",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes="btn-economico"
                                )
                                btn_ubicacion = gr.Button(
                                    "📍 ¿Dónde están ubicados?",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes="btn-ubicacion"
                                )

                            mensaje_usuario = gr.Textbox(
                                label="✍️ Escribe tu mensaje",
                                placeholder="Ejemplo: Necesito un celular para fotos con presupuesto de 2000 soles",
                                lines=2
                            )

                            with gr.Row():
                                btn_enviar = gr.Button(
                                    "🚀 Enviar",
                                    variant="primary",
                                    elem_classes="btn-enviar"
                                )
                                btn_limpiar = gr.Button(
                                    "🧹 Limpiar Chat",
                                    variant="secondary",
                                    elem_classes="btn-limpiar"
                                )

                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #0ea5e9;">
                                <h3 style="color: #0c4a6e; margin: 0 0 20px 0; font-weight: bold;">🎯 Funcionalidades</h3>
                                <ul style="color: #075985; margin: 0; padding-left: 20px;">
                                    <li style="margin-bottom: 8px;">📱 Recomendación inteligente de celulares</li>
                                    <li style="margin-bottom: 8px;">💰 Filtros avanzados por presupuesto</li>
                                    <li style="margin-bottom: 8px;">📸 Análisis especializado de cámaras</li>
                                    <li style="margin-bottom: 8px;">⚡ Evaluación de rendimiento</li>
                                    <li style="margin-bottom: 8px;">🔄 Respuestas estructuradas con IA</li>
                                    <li style="margin-bottom: 8px;">🏪 Información completa de MijoStore</li>
                                </ul>

                                <h3 style="color: #0c4a6e; margin: 25px 0 15px 0; font-weight: bold;">💡 Ejemplos de consultas:</h3>
                                <div style="background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px; font-size: 0.9em; color: #164e63;">
                                    <p style="margin: 5px 0;">• "Celular para gaming 3000 soles"</p>
                                    <p style="margin: 5px 0;">• "Mejor cámara presupuesto 1500"</p>
                                    <p style="margin: 5px 0;">• "¿Dónde están ubicados?"</p>
                                    <p style="margin: 5px 0;">• "Horarios de atención"</p>
                                    <p style="margin: 5px 0;">• "Redes sociales de la tienda"</p>
                                </div>
                            </div>
                            """)

                # Pestaña 2: Recomendación estructurada
                with gr.TabItem("🎯 Recomendación Avanzada"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #f59e0b; margin-bottom: 20px;">
                                <h3 style="color: #92400e; margin: 0 0 15px 0; font-weight: bold;">⚙️ Configuración de Búsqueda Avanzada</h3>
                                <p style="color: #a16207; margin: 0; font-size: 0.95em;">Personaliza tu búsqueda para encontrar el celular perfecto</p>
                            </div>
                            """)

                            presupuesto_input = gr.Slider(
                                minimum=500, maximum=6000, value=2000, step=100,
                                label="💰 Presupuesto Máximo (Soles)"
                            )

                            prioridad_camara = gr.Checkbox(
                                label="📸 Priorizar Calidad de Cámara",
                                value=False
                            )

                            prioridad_rendimiento = gr.Checkbox(
                                label="⚡ Priorizar Rendimiento",
                                value=False
                            )

                            uso_principal = gr.Dropdown(
                                choices=["Fotografía", "Gaming", "Trabajo", "Uso general", "Redes sociales"],
                                label="📋 Uso Principal",
                                value="Uso general"
                            )

                            marca_preferida = gr.Dropdown(
                                choices=["Sin preferencia", "Samsung", "iPhone", "Xiaomi", "Google", "OnePlus", "Realme"],
                                label="🏷️ Marca Preferida",
                                value="Sin preferencia"
                            )

                            btn_recomendar = gr.Button(
                                "🔍 Buscar Recomendación",
                                variant="primary",
                                elem_classes="btn-recomendar"
                            )

                        with gr.Column():
                            resultado_recomendacion = gr.Markdown(
                                label="📋 Resultado",
                                value="Configura tus preferencias y presiona 'Buscar Recomendación'"
                            )

                # Pestaña 3: Información de la Tienda
                with gr.TabItem("🏪 MijoStore"):
                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(255, 107, 53, 0.3);">
                        <h1 style="color: white; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🏪 MijoStore Tacna</h1>
                        <p style="color: white; font-size: 1.3em; margin: 15px 0 0 0; opacity: 0.95;">Tu tienda de confianza para celulares y tecnología</p>
                    </div>
                    """)

                    with gr.Row():
                        with gr.Column():
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #a855f7; margin-bottom: 20px;">
                                <h3 style="color: #7c2d92; margin: 0 0 20px 0; font-weight: bold;">📍 Nuestras Ubicaciones</h3>
                                <div style="color: #86198f;">
                                    <p><strong>🏪 Tienda Principal:</strong><br>Calle Zela Nro 267, Tacna</p>
                                    <p><strong>🏪 Sucursal:</strong><br>Cnel. Inclan 382-196, Tacna 23001</p>
                                </div>
                            </div>

                            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #22c55e; margin-bottom: 20px;">
                                <h3 style="color: #15803d; margin: 0 0 20px 0; font-weight: bold;">📞 Contáctanos</h3>
                                <div style="color: #166534;">
                                    <p><strong>📞 Teléfono:</strong> 052632704</p>
                                    <p><strong>📱 Celular/WhatsApp:</strong> +51952909892</p>
                                    <p><strong>📧 Email:</strong> mijostore.online@gmail.com</p>
                                </div>
                            </div>
                            """)

                        with gr.Column():
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #3b82f6; margin-bottom: 20px;">
                                <h3 style="color: #1d4ed8; margin: 0 0 20px 0; font-weight: bold;">🌐 Síguenos en Redes</h3>
                                <div style="color: #1e40af;">
                                    <p><strong>📘 Facebook:</strong><br><a href="https://www.facebook.com/mijostore.tacna" target="_blank" style="color: #3b82f6;">MijoStore Tacna</a></p>
                                    <p><strong>📸 Instagram:</strong><br><a href="https://www.instagram.com/mijostoretacna/?hl=es-la" target="_blank" style="color: #3b82f6;">@mijostoretacna</a></p>
                                    <p><strong>🌍 Sitio Web:</strong><br><a href="https://mijostore.pe" target="_blank" style="color: #3b82f6;">mijostore.pe</a></p>
                                </div>
                            </div>

                            <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #f59e0b; margin-bottom: 20px;">
                                <h3 style="color: #92400e; margin: 0 0 20px 0; font-weight: bold;">🕒 Horarios de Atención</h3>
                                <div style="color: #a16207;">
                                    <p><strong>Lunes a Sábado:</strong> 9:00 AM - 8:00 PM</p>
                                    <p><strong>Domingos:</strong> 10:00 AM - 6:00 PM</p>
                                </div>
                            </div>
                            """)

                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 30px; border-radius: 15px; text-align: center; border: 2px solid #ff6b35;">
                        <h3 style="color: #ff6b35; margin: 0 0 15px 0; font-weight: bold;">🛍️ Nuestra Especialidad</h3>
                        <p style="color: #475569; font-size: 1.1em; margin: 0;">Venta de celulares, accesorios y servicios técnicos especializados</p>
                        <div style="margin-top: 20px;">
                            <a href="https://maps.app.goo.gl/uc2KXhQr7bEHiZA86" target="_blank" style="background: #ff6b35; color: white; padding: 12px 25px; border-radius: 25px; text-decoration: none; margin: 5px; display: inline-block; font-weight: bold;">🗺️ Ver en Google Maps</a>
                            <a href="https://api.whatsapp.com/send/?phone=51952909892&text&type=phone_number&app_absent=0" target="_blank" style="background: #25d366; color: white; padding: 12px 25px; border-radius: 25px; text-decoration: none; margin: 5px; display: inline-block; font-weight: bold;">💬 WhatsApp Directo</a>
                        </div>
                    </div>
                    """)

                # Pestaña 4: Catálogo completo
                with gr.TabItem("📋 Catálogo"):
                    catalogo_df = self._crear_catalogo_dataframe()
                    gr.Dataframe(
                        value=catalogo_df,
                        label="📱 Catálogo de Celulares Disponibles",
                        interactive=False
                    )

            # Configurar eventos
            self._configurar_eventos(
                historial_chat, mensaje_usuario, btn_enviar, btn_limpiar,
                presupuesto_input, prioridad_camara, prioridad_rendimiento,
                uso_principal, marca_preferida, btn_recomendar, resultado_recomendacion,
                btn_gaming, btn_fotos, btn_economico, btn_ubicacion
            )

def _crear_catalogo_dataframe(self):
        """Crea un DataFrame con el catálogo de celulares"""
        celulares = self.chatbot_engine.base_datos.obtener_todos()
        data = []
        for c in celulares:
            data.append([
                c.marca, c.modelo, f"S/{c.precio:,.0f}",
                c.ram, c.almacenamiento, c.camara_principal,
                f"{c.puntuacion_foto}/10", f"{c.puntuacion_rendimiento}/10"
            ])

        return data


def _obtener_estilos_css(self) -> str:
        """
        Autor: Fabiola
        Estilos CSS personalizados para Mijito Store
        """
        return """
        /* Estilo general del contenedor */
        .gradio-container {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #fef7f0 0%, #fed7d7 50%, #f97316 100%);
            min-height: 100vh;
        }

        /* Estilo del chatbot */
        .chatbot {
            border-radius: 20px !important;
            box-shadow: 0 10px 40px rgba(255, 107, 53, 0.2) !important;
            border: 2px solid rgba(255, 107, 53, 0.1) !important;
            background: white !important;
        }

        /* Pestañas principales */
        .tab-nav {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            border-radius: 15px 15px 0 0 !important;
            box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3) !important;
        }

        .tab-nav button {
            color: white !important;
            font-weight: bold !important;
            background: transparent !important;
            border: none !important;
            padding: 15px 25px !important;
            transition: all 0.3s ease !important;
        }

        .tab-nav button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: translateY(-2px) !important;
        }

        .tab-nav button.selected {
            background: rgba(255, 255, 255, 0.3) !important;
            border-bottom: 3px solid white !important;
        }

        /* Botones principales */
        .btn-enviar {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 12px 30px !important;
            box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4) !important;
            transition: all 0.3s ease !important;
        }

        .btn-enviar:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(255, 107, 53, 0.6) !important;
        }

        .btn-recomendar {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 15px 35px !important;
            box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4) !important;
            transition: all 0.3s ease !important;
        }

        .btn-recomendar:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(245, 158, 11, 0.6) !important;
        }

        .btn-limpiar {
            background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 12px 30px !important;
            transition: all 0.3s ease !important;
        }

        .btn-limpiar:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 15px rgba(107, 114, 128, 0.4) !important;
        }

        /* Botones de preguntas rápidas */
        .btn-gaming {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        }

        .btn-gaming:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
        }

        .btn-fotos {
            background: linear-gradient(135deg, #ec4899 0%, #db2777 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3) !important;
        }

        .btn-fotos:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(236, 72, 153, 0.5) !important;
        }

        .btn-economico {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3) !important;
        }

        .btn-economico:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(34, 197, 94, 0.5) !important;
        }

        .btn-ubicacion {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
        }

        .btn-ubicacion:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
        }

        /* Campo de texto */
        .gr-textbox textarea {
            border-radius: 15px !important;
            border: 2px solid rgba(255, 107, 53, 0.2) !important;
            background: rgba(255, 255, 255, 0.9) !important;
            transition: all 0.3s ease !important;
        }

        .gr-textbox textarea:focus {
            border-color: #ff6b35 !important;
            box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.1) !important;
        }

        /* Sliders y controles */
        .gr-slider input[type="range"] {
            accent-color: #ff6b35 !important;
        }

        .gr-checkbox input[type="checkbox"]:checked {
            background-color: #ff6b35 !important;
            border-color: #ff6b35 !important;
        }

        .gr-dropdown .dropdown {
            border-radius: 15px !important;
            border: 2px solid rgba(255, 107, 53, 0.2) !important;
        }

        /* Animaciones suaves */
        * {
            transition: all 0.3s ease !important;
        }

        /* Efectos hover para tarjetas */
        div[style*="border-left: 5px solid"] {
            transition: all 0.3s ease !important;
        }

        div[style*="border-left: 5px solid"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
        }

        /* Estilo para el DataFrame del catálogo */
        .gr-dataframe {
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 6px 20px rgba(255, 107, 53, 0.15) !important;
        }

        .gr-dataframe th {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            color: white !important;
            font-weight: bold !important;
        }

        .gr-dataframe tr:nth-child(even) {
            background: rgba(255, 107, 53, 0.05) !important;
        }

        /* Mejoras de responsividad */
        @media (max-width: 768px) {
            .tab-nav button {
                padding: 10px 15px !important;
                font-size: 0.9em !important;
            }

            .btn-enviar, .btn-recomendar {
                padding: 10px 20px !important;
            }
        }
        """
def _configurar_eventos(self, historial, mensaje, btn_enviar, btn_limpiar,
                          presupuesto_input, prioridad_camara, prioridad_rendimiento,
                          uso_principal, marca_preferida, btn_recomendar, resultado_recomendacion,
                          btn_gaming, btn_fotos, btn_economico, btn_ubicacion) -> None:
        """
        Autor: Fabiola
        Configura los eventos de la interfaz
        """
        def procesar_mensaje(historial_actual, mensaje_usuario):
            if not mensaje_usuario.strip():
                return historial_actual, ""

            # Procesar mensaje con el motor del chatbot
            respuesta = self.chatbot_engine.procesar_mensaje(mensaje_usuario)

            # Update historial to match 'messages' type
            historial_actual.append({"role": "user", "content": mensaje_usuario})
            historial_actual.append({"role": "assistant", "content": respuesta})

            return historial_actual, ""

        def enviar_pregunta_rapida(pregunta, historial_actual):
            """Procesa una pregunta rápida predefinida"""
            return procesar_mensaje(historial_actual, pregunta)

        def procesar_recomendacion_estructurada(presupuesto, camara, rendimiento, uso, marca):
            """Procesa recomendación con parámetros estructurados"""
            try:
                consulta = ConsultaUsuario(
                    presupuesto_max=presupuesto,
                    prioridad_camara=camara,
                    prioridad_rendimiento=rendimiento,
                    uso_principal=uso,
                    marca_preferida=None if marca == "Sin preferencia" else marca
                )

                recomendacion = self.chatbot_engine.procesar_recomendacion_estructurada(consulta)

                # Formatear resultado de manera más amigable
                resultado = f"""## 🏆 **¡Tu Celular Perfecto!**

### 📱 **{recomendacion.celular_recomendado.marca} {recomendacion.celular_recomendado.modelo}**

💰 **Precio:** S/{recomendacion.celular_recomendado.precio:,.0f}
📸 **Cámara:** {recomendacion.celular_recomendado.camara_principal}
💾 **RAM:** {recomendacion.celular_recomendado.ram}
📱 **Pantalla:** {recomendacion.celular_recomendado.pantalla}
🔋 **Batería:** {recomendacion.celular_recomendado.bateria}

### ⭐ **Puntuaciones**
📸 **Fotos:** {recomendacion.celular_recomendado.puntuacion_foto}/10
⚡ **Rendimiento:** {recomendacion.celular_recomendado.puntuacion_rendimiento}/10

### 🤔 **¿Por qué te recomiendo este?**
{recomendacion.razonamiento}

### 📊 **Análisis**
✅ **Presupuesto:** {"✅ Dentro de tu presupuesto" if recomendacion.coincidencia_presupuesto else "⚠️ Ligeramente fuera del presupuesto"}
🎯 **Compatibilidad:** {recomendacion.puntuacion_match:.0%}

## 🔄 **Otras opciones geniales:**"""

                for i, alt in enumerate(recomendacion.alternativas, 1):
                    resultado += f"""

### {i}. **{alt.marca} {alt.modelo}**
💰 S/{alt.precio:,.0f} | 📸 {alt.camara_principal} | 💾 {alt.ram}"""

                resultado += f"""

### 🏪 **¡Disponibles en MijoStore!**
Todos estos celulares los tenemos en stock. ¿Te interesa alguno? ¡Contáctanos para más información! 📞"""

                return resultado

            except Exception as e:
                return f"❌ **Oops! Algo salió mal:** {str(e)}"

        def limpiar_chat():
            # Restaurar mensaje de bienvenida
            return [{
                "role": "assistant",
                "content": """🎉 **¡Hola! Soy Mijito, tu asistente virtual de MijoStore** 📱

¡Bienvenido! Estoy aquí para ayudarte a encontrar el celular perfecto según tus necesidades y presupuesto. Con tecnología de IA avanzada, puedo darte recomendaciones personalizadas.

### 🤖 ¿En qué puedo ayudarte hoy?
- 📱 Recomendarte el celular ideal para ti
- 💰 Filtrar opciones por tu presupuesto
- 📸 Encontrar los mejores para fotografía
- 🎮 Celulares perfectos para gaming
- 📍 Información sobre nuestras tiendas
- 📞 Datos de contacto y ubicación

### 💡 **¡Haz clic en una pregunta rápida para empezar!**"""
            }], ""

        # Eventos de botones principales
        btn_enviar.click(
            fn=procesar_mensaje,
            inputs=[historial, mensaje],
            outputs=[historial, mensaje]
        )

        mensaje.submit(
            fn=procesar_mensaje,
            inputs=[historial, mensaje],
            outputs=[historial, mensaje]
        )

        btn_limpiar.click(
            fn=limpiar_chat,
            outputs=[historial, mensaje]
        )

        # Eventos de preguntas rápidas
        btn_gaming.click(
            fn=lambda hist: enviar_pregunta_rapida("Necesito un celular para gaming con presupuesto de 3000 soles", hist),
            inputs=[historial],
            outputs=[historial, mensaje]
        )

        btn_fotos.click(
            fn=lambda hist: enviar_pregunta_rapida("Busco el celular con mejor cámara por 2000 soles", hist),
            inputs=[historial],
            outputs=[historial, mensaje]
        )

        btn_economico.click(
            fn=lambda hist: enviar_pregunta_rapida("¿Cuál es el mejor celular económico por 1500 soles?", hist),
            inputs=[historial],
            outputs=[historial, mensaje]
        )

        btn_ubicacion.click(
            fn=lambda hist: enviar_pregunta_rapida("¿Dónde están ubicados?", hist),
            inputs=[historial],
            outputs=[historial, mensaje]
        )

        btn_recomendar.click(
            fn=procesar_recomendacion_estructurada,
            inputs=[presupuesto_input, prioridad_camara, prioridad_rendimiento, uso_principal, marca_preferida],
            outputs=[resultado_recomendacion]
        )

    def lanzar(self, compartir: bool = True) -> None:
        """
        Autor: Fabiola
        Lanza la interfaz web
        """
        self.app.launch(
            share=compartir,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
# 🚀 Clase Principal del Sistema
class CelularBotSystem:
    """
    Autor: Erick
    Clase principal que coordina todo el sistema de recomendación
    """

    def __init__(self):
        self.config = ConfiguracionSistema()
        self.servidor_llm = None
        self.chatbot_engine = None
        self.interfaz_web = None

    def inicializar_sistema(self) -> None:
        """
        Autor: Erick
        Inicializa todos los componentes del sistema
        """
        print("🚀 Inicializando CelularBot System...")

        # 1. Inicializar servidor LLM
        print("📡 Configurando servidor LLM...")
        self.servidor_llm = ServidorLLM(self.config)
        self.servidor_llm.iniciar_servidor()

        # Esperar a que el servidor esté listo
        time.sleep(5)

        # 2. Inicializar motor del chatbot con base de datos
        print("🧠 Configurando motor del chatbot...")
        self.chatbot_engine = ChatbotEngine(self.config)
        print(f"📱 Base de datos cargada con {len(self.chatbot_engine.base_datos.obtener_todos())} celulares")

        # 3. Crear interfaz web
        print("🎨 Creando interfaz web...")
        self.interfaz_web = InterfazWeb(self.chatbot_engine)

        print("✅ Sistema inicializado correctamente!")

    def ejecutar(self) -> None:
        """
        Autor: Erick
        Ejecuta el sistema completo
        """
        self.inicializar_sistema()

        print("🌐 Lanzando interfaz web...")
        print("🔗 El recomendador de celulares estará disponible en la URL que se muestre a continuación")

        self.interfaz_web.lanzar(compartir=True)

# 🎯 Ejecución Principal
if __name__ == "__main__":
    """
    Autor: Equipo Completo
    Punto de entrada principal del sistema
    """
    print("=" * 70)
    print("📱 CELULARBOT - SISTEMA DE RECOMENDACIÓN INTELIGENTE")
    print("📚 Proyecto Universitario - LangChain + IA")
    print("👥 Equipo de Desarrollo: Erick, Dylan, Fabiola")
    print("=" * 70)
    print()
    print("🎯 CARACTERÍSTICAS IMPLEMENTADAS:")
    print("✅ Cadenas LangChain (LCEL)")
    print("✅ Runnable Functions")
    print("✅ Cadenas Paralelas")
    print("✅ Chain of Thought (CoT)")
    print("✅ Salida Estructurada (Pydantic)")
    print("✅ Mensajes Sistema/AI/Humano")
    print("✅ Base de datos de celulares")
    print("✅ Interfaz web avanzada")
    print("=" * 70)

    try:
        # Crear e inicializar el sistema
        sistema = CelularBotSystem()
        sistema.ejecutar()

    except KeyboardInterrupt:
        print("\n⛔ Sistema detenido por el usuario")
    except Exception as e:
        print(f"❌ Error crítico en el sistema: {e}")
        raise