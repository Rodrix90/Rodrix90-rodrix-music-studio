import os
import re
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from audiocraft.models import MusicGen


# ============================================================
# RODRIX MUSIC CREATOR
# Gradio + MusicGen + fondos de video 4K online
# ============================================================

APP_TITLE = "Rodrix Music Creator"

# Videos 4K online. No se guardan dentro del proyecto.
# Podés reemplazarlos sin tocar el código definiendo una variable de entorno:
# BACKGROUND_VIDEOS="https://...mp4,https://...mp4"
DEFAULT_BACKGROUND_VIDEOS = [
    # Acantilados del Mar del Norte - 3840x2160 / 60 fps
    "https://videos.pexels.com/video-files/36505810/15479717_3840_2160_60fps.mp4",
    # Playa / naturaleza - 3840x2160 / 25 fps
    "https://videos.pexels.com/video-files/9140346/9140346-uhd_3840_2160_25fps.mp4",
]


def obtener_videos_fondo():
    configurados = os.environ.get("BACKGROUND_VIDEOS", "").strip()
    if configurados:
        videos = [url.strip() for url in configurados.split(",") if url.strip()]
        if videos:
            return videos
    return DEFAULT_BACKGROUND_VIDEOS


BACKGROUND_VIDEOS = obtener_videos_fondo()


# ============================================================
# MODELO IA
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

try:
    print("Cargando modelo de IA...")
    model = MusicGen.get_pretrained("facebook/musicgen-melody", device=device)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


# ============================================================
# UTILIDADES
# ============================================================

def sanear_nombre_archivo(nombre: str) -> str:
    nombre = (nombre or "").strip()
    if not nombre:
        return "Himno_Generado"

    nombre = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñ._-]+", "_", nombre)
    nombre = nombre.strip("._-")
    return nombre or "Himno_Generado"


def convertir_a_mp3(wav_path: str, mp3_path: str) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-b:a",
                "192k",
                mp3_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise gr.Error("FFmpeg no está instalado o no está disponible en el PATH.")
    except subprocess.CalledProcessError as e:
        detalle = (e.stderr or "").strip()
        raise gr.Error(f"FFmpeg no pudo generar el MP3. {detalle[-500:]}")


# ============================================================
# GENERACIÓN DE AUDIO
# ============================================================

def procesar_audio(
    himno_audio,
    letras_texto,
    estilo_prompt,
    titulo_cancion,
    excluir_estilos,
    genero_vocal,
    weirdness,
    style_influence,
    audio_influence,
    formato_salida,
    historial_archivos,
):
    if himno_audio is None:
        raise gr.Error("Por favor, sube un archivo de audio (Himno base).")

    if model is None:
        raise gr.Error("El modelo no se cargó correctamente. Revisa los logs.")

    sr, audio_data = himno_audio

    num_samples = audio_data.shape[0] if len(audio_data.shape) > 1 else audio_data.shape[-1]
    duracion_real = num_samples / sr
    duracion_final = min(duracion_real, 30.0)

    if duracion_final <= 0:
        raise gr.Error("El archivo de audio no tiene una duración válida.")

    # Weirdness 0-100 -> Temperature 1.0-1.5
    temp = 1.0 + (weirdness / 200.0)

    # Style Influence 0-100 -> CFG 2.0-8.0
    cfg = 2.0 + (style_influence / 100.0) * 6.0

    model.set_generation_params(
        duration=duracion_final,
        top_k=250,
        top_p=0.0,
        temperature=temp,
        cfg_coef=cfg,
    )

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    # Audio Influence 0-100
    vol_multiplier = max(0.1, audio_influence / 50.0)
    audio_data = audio_data * vol_multiplier

    melody = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
    model_sr = model.sample_rate

    if sr != model_sr:
        melody = torchaudio.functional.resample(
            melody,
            orig_freq=sr,
            new_freq=model_sr,
        )

    prompt_completo = (estilo_prompt or "").strip()

    if genero_vocal == "Male":
        prompt_completo += ", male vocals, male singer"
    elif genero_vocal == "Female":
        prompt_completo += ", female vocals, female singer"

    excluir_estilos = (excluir_estilos or "").strip()
    if excluir_estilos:
        prompt_completo += f". Do not use: {excluir_estilos}. Without {excluir_estilos}"

    letras_texto = (letras_texto or "").strip()
    if letras_texto:
        prompt_completo = f"{letras_texto}. {prompt_completo}"

    prompt_completo = prompt_completo.strip(" ,.")
    if not prompt_completo:
        prompt_completo = "instrumental music"

    print(f"Generando audio con prompt: '{prompt_completo}'...")

    try:
        generacion = model.generate_with_chroma(
            descriptions=[prompt_completo],
            melody_wavs=melody,
            melody_sample_rate=model_sr,
            progress=True,
        )
    except Exception as e:
        raise gr.Error(f"Error durante la generación: {e}")

    salida_sr = 48000
    audio_generado = generacion[0].cpu()

    if model_sr != salida_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=model_sr,
            new_freq=salida_sr,
        )
        audio_hq = resampler(audio_generado)
    else:
        audio_hq = audio_generado

    titulo_saneado = sanear_nombre_archivo(titulo_cancion)

    carpeta_tmp = Path("/tmp")
    carpeta_tmp.mkdir(parents=True, exist_ok=True)

    wav_path = str(carpeta_tmp / f"{titulo_saneado}.wav")

    audio_numpy = audio_hq.numpy()

    if audio_numpy.shape[0] == 1:
        audio_numpy = np.repeat(audio_numpy, 2, axis=0)

    audio_numpy = audio_numpy.T
    sf.write(wav_path, audio_numpy, salida_sr, subtype="PCM_24")

    if formato_salida == "mp3":
        mp3_path = str(carpeta_tmp / f"{titulo_saneado}.mp3")
        convertir_a_mp3(wav_path, mp3_path)
        output_path = mp3_path
    else:
        output_path = wav_path

    if historial_archivos is None:
        historial_archivos = []

    historial_archivos = list(historial_archivos)
    historial_archivos.append(output_path)

    return output_path, historial_archivos, historial_archivos


# ============================================================
# FONDO 4K ONLINE
# ============================================================

# El navegador reproduce solo pequeños segmentos de cada video.
# Se cambia al siguiente cada 10 segundos.
VIDEO_SEGMENT_SECONDS = 10

videos_js = ",\n".join(f'        "{url}"' for url in BACKGROUND_VIDEOS)

CUSTOM_HEAD = f"""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>
(() => {{
    const RODRIX_VIDEOS = [
{videos_js}
    ];

    const SEGMENT_SECONDS = {VIDEO_SEGMENT_SECONDS};
    let currentIndex = Math.floor(Math.random() * RODRIX_VIDEOS.length);
    let changeTimer = null;

    function createBackground() {{
        if (document.getElementById("rodrix-bg-video")) return;
        if (!RODRIX_VIDEOS.length) return;

        const video = document.createElement("video");
        video.id = "rodrix-bg-video";
        video.autoplay = true;
        video.muted = true;
        video.loop = false;
        video.playsInline = true;
        video.preload = "metadata";
        video.setAttribute("muted", "");
        video.setAttribute("playsinline", "");
        video.setAttribute("aria-hidden", "true");

        const overlay = document.createElement("div");
        overlay.id = "rodrix-bg-overlay";
        overlay.setAttribute("aria-hidden", "true");

        document.body.prepend(overlay);
        document.body.prepend(video);

        function playCurrent() {{
            video.classList.remove("rodrix-video-visible");
            video.src = RODRIX_VIDEOS[currentIndex];
            video.currentTime = 0;
            video.load();

            const startPlayback = () => {{
                video.muted = true;
                video.play().catch(() => {{}});
                requestAnimationFrame(() => {{
                    video.classList.add("rodrix-video-visible");
                }});

                clearTimeout(changeTimer);
                changeTimer = setTimeout(nextVideo, SEGMENT_SECONDS * 1000);
            }};

            if (video.readyState >= 2) {{
                startPlayback();
            }} else {{
                video.addEventListener("canplay", startPlayback, {{ once: true }});
            }}
        }}

        function nextVideo() {{
            video.classList.remove("rodrix-video-visible");

            setTimeout(() => {{
                currentIndex = (currentIndex + 1) % RODRIX_VIDEOS.length;
                playCurrent();
            }}, 450);
        }}

        video.addEventListener("ended", nextVideo);
        video.addEventListener("error", () => {{
            clearTimeout(changeTimer);
            currentIndex = (currentIndex + 1) % RODRIX_VIDEOS.length;
            setTimeout(playCurrent, 1000);
        }});

        playCurrent();
    }}

    document.addEventListener("DOMContentLoaded", createBackground);
    window.addEventListener("load", createBackground);
    setTimeout(createBackground, 600);
}})();
</script>
"""


CUSTOM_CSS = """
html, body {
    min-height: 100%;
    background: #05070a !important;
}

body {
    margin: 0 !important;
    overflow-x: hidden;
}

#rodrix-bg-video {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    object-fit: cover;
    object-position: center center;
    z-index: -30;
    opacity: 0;
    transition: opacity 0.45s ease;
    pointer-events: none;
    background: #05070a;
    transform: translateZ(0);
}

#rodrix-bg-video.rodrix-video-visible {
    opacity: 1;
}

#rodrix-bg-overlay {
    position: fixed;
    inset: 0;
    z-index: -20;
    pointer-events: none;
    background:
        radial-gradient(circle at 70% 12%, rgba(255,255,255,0.07), transparent 34%),
        linear-gradient(180deg, rgba(3,5,8,0.42) 0%, rgba(3,5,8,0.66) 52%, rgba(3,5,8,0.88) 100%);
}

.gradio-container {
    background: transparent !important;
    min-height: 100vh !important;
}

#rodrix-shell {
    max-width: 1450px;
    margin: 0 auto;
    padding-top: 20px;
}

#rodrix-header {
    margin-bottom: 14px;
    padding: 18px 22px;
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 22px;
    background: rgba(10, 12, 16, 0.56);
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

#rodrix-header h1 {
    margin: 0;
    font-size: clamp(28px, 4vw, 48px);
    letter-spacing: -0.04em;
    color: white;
}

#rodrix-header p {
    margin: 6px 0 0 0;
    color: rgba(255,255,255,0.72);
}

.rodrix-panel {
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 22px !important;
    background: rgba(10, 12, 16, 0.62) !important;
    box-shadow: 0 18px 48px rgba(0,0,0,0.28) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    padding: 18px !important;
}

.rodrix-panel .block,
.rodrix-panel .form,
.rodrix-panel fieldset {
    background: rgba(18, 20, 25, 0.78) !important;
}

button.primary {
    font-weight: 800 !important;
}

footer {
    display: none !important;
}

@media (max-width: 800px) {
    #rodrix-shell {
        padding-top: 8px;
    }

    #rodrix-header {
        border-radius: 16px;
        padding: 15px 16px;
    }

    .rodrix-panel {
        border-radius: 16px !important;
        padding: 12px !important;
    }

    #rodrix-bg-video {
        object-position: center center;
    }
}

@media (prefers-reduced-motion: reduce) {
    #rodrix-bg-video {
        display: none !important;
    }
}
"""


# ============================================================
# INTERFAZ GRADIO
# ============================================================

usuario_valido = os.environ.get("APP_USER", "admin")
password_valida = os.environ.get("APP_PASS", "secreto123")


tema_suno = gr.themes.Monochrome(
    primary_hue="neutral",
    secondary_hue="neutral",
    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_50",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    button_primary_background_fill="*neutral_100",
    button_primary_text_color="*neutral_950",
)


GRADIO_MAJOR = int(str(getattr(gr, "__version__", "5")).split(".")[0])

# Gradio 6 movió theme/css/head desde Blocks() hacia launch().
# Este bloque mantiene compatibilidad con Gradio 4/5 y 6+.
if GRADIO_MAJOR >= 6:
    blocks_kwargs = {
        "title": APP_TITLE,
    }
else:
    blocks_kwargs = {
        "theme": tema_suno,
        "title": APP_TITLE,
        "css": CUSTOM_CSS,
        "head": CUSTOM_HEAD,
    }


with gr.Blocks(**blocks_kwargs) as demo:
    historial_state = gr.State([])

    with gr.Column(elem_id="rodrix-shell"):
        gr.HTML(
            """
            <div id="rodrix-header">
                <h1>🎵 Rodrix Music Creator</h1>
                <p>Creación musical asistida por IA · Fondo dinámico 4K online</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes=["rodrix-panel"]):
                gr.Markdown("### 🎼 Crear")

                himno_input = gr.Audio(
                    label="+ Audio · Sube tu melodía base",
                    type="numpy",
                )

                letras_input = gr.Textbox(
                    label="Letra",
                    placeholder="Escribe la letra aquí...",
                    lines=4,
                )

                estilo_input = gr.Textbox(
                    label="Estilo",
                    placeholder="Ej. acoustic folk, slow waltz, latin pop...",
                    lines=2,
                )

                with gr.Accordion("Más opciones", open=False):
                    excluir_input = gr.Textbox(
                        label="Excluir estilos",
                        placeholder="Ej. rock, electronic, loud drums",
                    )

                    genero_vocal = gr.Radio(
                        ["Random", "Male", "Female"],
                        label="Género vocal",
                        value="Random",
                    )

                    weirdness_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Weirdness (%)",
                    )

                    style_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Influencia de estilo (%)",
                    )

                    audio_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Influencia del audio (%)",
                    )

                with gr.Row():
                    titulo_input = gr.Textbox(
                        label="Título de la canción",
                        placeholder="Mi Nueva Canción",
                    )

                    formato_salida = gr.Radio(
                        ["wav", "mp3"],
                        label="Formato",
                        value="wav",
                    )

                with gr.Row():
                    btn_clear = gr.Button("🗑️ Limpiar", size="sm")
                    btn_generar = gr.Button("✨ Crear canción", variant="primary")

            with gr.Column(scale=2, elem_classes=["rodrix-panel"]):
                gr.Markdown("### 📚 Biblioteca / Resultado")

                resultado_audio = gr.Audio(
                    label="Canción actual · Alta fidelidad",
                    type="filepath",
                )

                gr.Markdown("#### Canciones generadas en esta sesión")

                libreria_archivos = gr.Files(
                    label="Historial de descargas"
                )

    def limpiar():
        return None, "", "", "", "", "Random", 50, 50, 50, "wav"

    btn_clear.click(
        fn=limpiar,
        inputs=[],
        outputs=[
            himno_input,
            letras_input,
            estilo_input,
            titulo_input,
            excluir_input,
            genero_vocal,
            weirdness_slider,
            style_slider,
            audio_slider,
            formato_salida,
        ],
    )

    btn_generar.click(
        fn=procesar_audio,
        inputs=[
            himno_input,
            letras_input,
            estilo_input,
            titulo_input,
            excluir_input,
            genero_vocal,
            weirdness_slider,
            style_slider,
            audio_slider,
            formato_salida,
            historial_state,
        ],
        outputs=[
            resultado_audio,
            libreria_archivos,
            historial_state,
        ],
    )


# ============================================================
# ARRANQUE
# ============================================================

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "7860"))

    print(f"Iniciando {APP_TITLE} en puerto {puerto}")
    print(f"Fondos 4K configurados: {len(BACKGROUND_VIDEOS)}")

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": puerto,
        "auth": (usuario_valido, password_valida),
    }

    if GRADIO_MAJOR >= 6:
        launch_kwargs.update(
            theme=tema_suno,
            css=CUSTOM_CSS,
            head=CUSTOM_HEAD,
        )

    demo.launch(**launch_kwargs)
