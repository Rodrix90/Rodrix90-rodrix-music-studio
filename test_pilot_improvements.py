import os
import sys
import wave
import struct
import math
import io
import asyncio
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import app from app.py
from app import app, TASKS, RECENT_OPERATIONS, get_current_user

def create_dummy_wav(duration_sec=2, sample_rate=44100, freq=440):
    """Creates a valid in-memory PCM WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        raw_data = bytearray()
        for i in range(num_frames):
            val = int(10000 * math.sin(2 * math.pi * freq * i / sample_rate))
            raw_data.extend(struct.pack('<h', val))
        wf.writeframes(raw_data)
    buf.seek(0)
    return buf.read()

def test_suite():
    print("=== INICIANDO VERIFICACIONES FINALES EXHAUSTIVAS (100% OFFLINE) ===")
    
    # Bypass auth for testing
    from app import get_current_user
    app.dependency_overrides[get_current_user] = lambda: "test_rodrigo"

    # Tests 1-3 use mock for background transform tasks
    with patch("app.run_transform_task", new=MagicMock()):
        client = TestClient(app)

        # ---------------------------------------------------------------------
        # 1. TEST: Prevención de duplicados con contenido real
        # ---------------------------------------------------------------------
        print("\n[1] Probando que dos archivos distintos de igual duración NO se confundan...")
        wav_440 = create_dummy_wav(duration_sec=2, freq=440)
        wav_880 = create_dummy_wav(duration_sec=2, freq=880)
        assert len(wav_440) == len(wav_880), "Deben tener el mismo tamaño en bytes"
        assert wav_440 != wav_880, "Pero contenido espectral diferente"

        # Archivo 1 sin client_operation_id
        r_f1 = client.post(
            "/api/transform",
            data={"style": "worship piano", "lyrics": "Himno A", "title": "Mismo Titulo", "audio_influence": "50", "style_influence": "50", "weirdness": "25", "model": "chirp-v5"},
            files={"audio": ("audio_a.wav", io.BytesIO(wav_440), "audio/wav")}
        )
        task_id_1 = r_f1.json()["task_id"]

        # Archivo 2 con mismo título, estilo, modelo y duración en bytes, pero diferente audio
        r_f2 = client.post(
            "/api/transform",
            data={"style": "worship piano", "lyrics": "Himno A", "title": "Mismo Titulo", "audio_influence": "50", "style_influence": "50", "weirdness": "25", "model": "chirp-v5"},
            files={"audio": ("audio_b.wav", io.BytesIO(wav_880), "audio/wav")}
        )
        task_id_2 = r_f2.json()["task_id"]

        assert task_id_1 != task_id_2, "Dos archivos con diferente contenido NO deben colisionar como duplicados"
        print(f"   -> ÉXITO: Archivo A (Task: {task_id_1[:8]}...) y Archivo B (Task: {task_id_2[:8]}...) procesados independientemente gracias al hash de contenido.")

        # Re-enviando Archivo 1 con client_operation_id
        r_dup1 = client.post(
            "/api/transform",
            data={"style": "worship", "lyrics": "H", "title": "T", "client_operation_id": "op_unique_99", "model": "chirp-v5"},
            files={"audio": ("a.wav", io.BytesIO(wav_440), "audio/wav")}
        )
        r_dup2 = client.post(
            "/api/transform",
            data={"style": "worship", "lyrics": "H", "title": "T", "client_operation_id": "op_unique_99", "model": "chirp-v5"},
            files={"audio": ("a.wav", io.BytesIO(wav_440), "audio/wav")}
        )
        assert r_dup2.json().get("duplicate_prevented") is True
        assert r_dup2.json()["task_id"] == r_dup1.json()["task_id"]
        print("   -> ÉXITO: Operación idéntica detectada y reusada por client_operation_id.")

        # ---------------------------------------------------------------------
        # 2. TEST: Limpieza de archivos temporales en validación FFprobe
        # ---------------------------------------------------------------------
        print("\n[2] Comprobando que FFprobe no deja archivos temporales huérfanos...")
        tmp_before = set(os.listdir(tempfile.gettempdir()))
        _ = client.post(
            "/api/transform",
            data={"style": "worship", "lyrics": "H", "title": "T", "model": "chirp-v5"},
            files={"audio": ("sample.wav", io.BytesIO(wav_440), "audio/wav")}
        )
        tmp_after = set(os.listdir(tempfile.gettempdir()))
        new_tmp_files = [f for f in tmp_after - tmp_before if f.startswith("rodrix_val_")]
        print(f"   -> Archivos temporales de validación no limpiados: {len(new_tmp_files)}")
        assert len(new_tmp_files) == 0, f"Quedaron archivos temporales: {new_tmp_files}"
        print("   -> ÉXITO: Bloque finally ejecutó la limpieza al 100%.")

        # ---------------------------------------------------------------------
        # 3. TEST: Prueba funcional real de Fase 1 (restore_hymn_audio con FFmpeg)
        # ---------------------------------------------------------------------
        print("\n[3] Ejecutando prueba funcional real de Fase 1 (/api/hymn_production/restore)...")
        r_fase1 = client.post(
            "/api/hymn_production/restore",
            data={
                "title": "Himno Test Funcional",
                "intensity": "medium",
                "denoise_db": "10.0",
                "debox_cut": "true",
                "clarity_boost": "true",
                "air_shelf": "true",
                "compression_profile": "smooth",
                "target_lufs": "-14.0"
            },
            files={"audio_file": ("grabacion_celular.wav", io.BytesIO(wav_440), "audio/wav")}
        )
        assert r_fase1.status_code == 200, f"Error Fase 1: {r_fase1.text}"
        res_fase1 = r_fase1.json()
        assert res_fase1.get("success") is True
        assert "processed_metrics" in res_fase1
        assert "original_metrics" in res_fase1
        print(f"   -> Masterización completada: LUFS Medido = {res_fase1['processed_metrics']['integrated_lufs']} LUFS, True Peak = {res_fase1['processed_metrics']['true_peak_db']} dBFS")
        print("   -> ÉXITO: Fase 1 opera íntegramente de extremo a extremo sin regresiones.")

        # ---------------------------------------------------------------------
        # 4. TEST: Distinción de errores (Catálogo vs Moderación vs Créditos vs Técnico)
        # ---------------------------------------------------------------------
        print("\n[4] Probando distinción rigurosa de errores y diagnósticos...")
        
        # Caso 4A: Coincidencia de catálogo
        TASKS["task_cat"] = {
            "id": "task_cat", "status": "ERROR", "title": "Himno Cat", "catalog_rejection": True, "detail": "Catalogo",
            "diagnostic_info": {"task_id": "task_cat", "error_type": "CATALOG_MATCH", "exact_error_message": "matches an existing recording in our catalog"}
        }
        r_rep_cat = client.get("/api/diagnostic_report/task_cat")
        assert "COINCIDENCIA DE" in r_rep_cat.text
        assert "support@udioapi.pro" in r_rep_cat.text

        # Caso 4B: Moderación de contenido / letra
        TASKS["task_mod"] = {
            "id": "task_mod", "status": "ERROR", "title": "Himno Mod", "catalog_rejection": False, "detail": "Moderacion",
            "diagnostic_info": {"task_id": "task_mod", "error_type": "CONTENT_MODERATION", "exact_error_message": "content policy violation: flagged lyrics"}
        }
        r_rep_mod = client.get("/api/diagnostic_report/task_mod")
        assert "MODERACIÓN DE CONTENIDO / POLÍTICA DE TEXTO" in r_rep_mod.text
        assert "Revise la letra y reemplace palabras" in r_rep_mod.text
        assert "COINCIDENCIA DE CATÁLOGO" not in r_rep_mod.text

        # Caso 4C: Error técnico genérico
        TASKS["task_tech"] = {
            "id": "task_tech", "status": "ERROR", "title": "Himno Tech", "catalog_rejection": False, "detail": "Fallo",
            "diagnostic_info": {"task_id": "task_tech", "error_type": "GENERIC", "exact_error_message": "500 Internal Server Error in audio worker"}
        }
        r_rep_tech = client.get("/api/diagnostic_report/task_tech")
        assert "ESTADO DE TAREA: ERROR" in r_rep_tech.text
        assert "500 Internal Server Error" in r_rep_tech.text
        assert "COINCIDENCIA DE" not in r_rep_tech.text
        print("   -> ÉXITO: Los tres tipos de error están nítidamente clasificados en la UI y en los informes TXT.")

        # ---------------------------------------------------------------------
        # 5. TEST: Preservación de valores predeterminados y presets
        # ---------------------------------------------------------------------
        print("\n[5] Verificando presets y selectores en index.html...")
        with open("index.html", "r", encoding="utf-8") as f:
            html = f.read()
        assert '<option value="chirp-v5" selected>' in html, "El modelo por defecto debe seguir siendo chirp-v5"
        assert 'id="weirdness" min="0" max="100" value="25"' in html, "Weirdness por defecto debe seguir siendo 25"
        assert 'id="styleInfluence" min="0" max="100" value="50"' in html, "Style influence por defecto debe ser 50"
        assert 'id="audioInfluence" min="0" max="100" value="25"' in html, "Audio influence por defecto debe ser 25"
        assert 'dw collector\'s maple' in html.lower(), "Preset DW debe seguir intacto"
        assert 'fender american deluxe jazz bass' in html.lower(), "Preset Fender Jazz Bass debe seguir intacto"
        print("   -> ÉXITO: Presets de instrumentos, modelos y valores predeterminados 100% inalterados.")

    # ---------------------------------------------------------------------
    # 6. TEST: Verificación de las 3 familias musicales y exclusiones selectivas
    # ---------------------------------------------------------------------
    print("\n[6] Probando construcción de estilos para las 3 familias y presets especiales...")
    from app import run_transform_task

    captured_payloads = []
    async def mock_post(client_self, url, json=None, headers=None, **kwargs):
        if json:
            captured_payloads.append(json)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"workId": "test_mock_work_id"}'
        mock_resp.json.return_value = {"workId": "test_mock_work_id"}
        return mock_resp

    async def mock_get(client_self, url, headers=None, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "SUCCESS"}'
        mock_resp.json.return_value = {
            "status": "SUCCESS",
            "data": [{
                "id": "trk_1", "title": "Test", "audio_url": "https://example.com/audio.mp3",
                "status": "SUCCESS", "lyrics": "Test"
            }]
        }
        return mock_resp

    async def run_family_tests():
        with patch("httpx.AsyncClient.post", new=mock_post), \
             patch("httpx.AsyncClient.get", new=mock_get):
            
            # Caso A: Himno Tradicional con banda seleccionada por el usuario (DW + Fender)
            TASKS["t_hymn"] = {"id": "t_hymn", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_hymn",
                style="himno tradicional, 72 bpm, dw collector's maple, fender jazz bass",
                lyrics="Sublime gracia del Señor",
                title="Himno Tradicional Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_hymn = captured_payloads[-1]
            style_hymn = payload_hymn["style"].lower()
            assert "sacred traditional christian hymn" in style_hymn
            assert "72.0 bpm" in style_hymn
            assert "dw collector" in style_hymn
            assert "fender jazz bass" in style_hymn
            assert "orchestral strings" not in style_hymn
            assert "no cumbia" not in payload_hymn.get("negative_tags", "")

            # Caso B: Coro Congregacional a tempo propio (105 BPM)
            TASKS["t_coro"] = {"id": "t_coro", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_coro",
                style="coro congregacional, 105 bpm, guitarra acustica",
                lyrics="Alabare, alabare",
                title="Coro Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="female",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_coro = captured_payloads[-1]
            style_coro = payload_coro["style"].lower()
            assert "congregational christian praise" in style_coro
            assert "105.0 bpm" in style_coro
            assert "118 bpm" not in style_coro  # No forzar 118 BPM

            # Caso C: Composición Moderna sin etiquetas religiosas forzadas
            TASKS["t_modern"] = {"id": "t_modern", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_modern",
                style="cancion original pop rock, 120 bpm, yamaha c5, neve 1073",
                lyrics="Caminando por la ciudad",
                title="Moderna Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=20,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_modern = captured_payloads[-1]
            style_modern = payload_modern["style"].lower()
            assert "contemporary studio production" in style_modern
            assert "120.0 bpm" in style_modern
            assert "yamaha c5" in style_modern
            assert "neve 1073" in style_modern

            # Caso D: Preset especial Vals Vienés 3/4
            TASKS["t_vals"] = {"id": "t_vals", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_vals",
                style="vals vienes, 3/4 time",
                lyrics="Santo Santo",
                title="Vals Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="metal",
                vocal_gender="random",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_vals = captured_payloads[-1]
            style_vals = payload_vals["style"].lower()
            assert "sacred viennese waltz hymn in 3/4 time" in style_vals

            # Caso E: Coro Congregacional en compás 3/4 (avivamiento sin imponer 4/4 ni gospel 2-step)
            TASKS["t_coro_34"] = {"id": "t_coro_34", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_coro_34",
                style="coro congregacional, avivamiento, compas 3/4, 96 bpm, guitarra acustica",
                lyrics="Grande y sublime es el Señor",
                title="Coro 3/4 Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_coro_34 = captured_payloads[-1]
            style_coro_34 = payload_coro_34["style"].lower()
            assert "congregational christian praise" in style_coro_34
            assert "96.0 bpm" in style_coro_34
            assert "compas 3/4" in style_coro_34
            assert "gospel 2-step" not in style_coro_34
            assert "straight 4/4" not in style_coro_34

            # Caso F: Composición moderna pop rock que menciona poéticamente la palabra "himno"
            TASKS["t_modern_himno"] = {"id": "t_modern_himno", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_modern_himno",
                style="estilo moderno, pop rock, un himno a la esperanza, 125 bpm, fender stratocaster",
                lyrics="Esta es una cancion de luz",
                title="Himno Pop Rock Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="female",
                weirdness=20,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_modern_himno = captured_payloads[-1]
            style_modern_himno = payload_modern_himno["style"].lower()
            assert "contemporary studio production" in style_modern_himno
            assert "sacred traditional christian hymn" not in style_modern_himno
            assert "125.0 bpm" in style_modern_himno

            # Caso G: Canción con voz solista sin coro
            TASKS["t_solo_voice"] = {"id": "t_solo_voice", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_solo_voice",
                style="balada acustica, voz solista, sin coro, 75 bpm, piano yamaha",
                lyrics="Solo ante ti mi Dios",
                title="Voz Solista Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_solo_voice = captured_payloads[-1]
            style_solo_voice = payload_solo_voice["style"].lower()
            prompt_solo_voice = payload_solo_voice.get("prompt", "")
            assert "no choir" in style_solo_voice or "sin coro" in style_solo_voice
            assert "church choir" not in style_solo_voice
            assert "male" in style_solo_voice
            # Letra 100% limpia de instrucciones metatextuales:
            assert "[Vocals:" not in prompt_solo_voice
            assert "[Language:" not in prompt_solo_voice
            assert prompt_solo_voice == "Solo ante ti mi Dios"

            # Caso H: Comprobación de que clean_lyrics_text preserva estrofas y etiquetas estructurales intactas
            from app import clean_lyrics_text, obfuscate_lyrics
            test_lyrics_with_structure = "[Verse 1]\nSublime gracia del Señor\n\n[Chorus]\nAleluya, gloria a Dios\n"
            cleaned_result = clean_lyrics_text(test_lyrics_with_structure)
            assert "[Verse 1]" in cleaned_result
            assert "[Chorus]" in cleaned_result
            assert "Sublime gracia del Señor" in cleaned_result
            assert "Aleluya, gloria a Dios" in cleaned_result
            # Comprobación de alias de compatibilidad hacia atrás
            assert obfuscate_lyrics(test_lyrics_with_structure) == cleaned_result

            # Caso I: Comprobación de patrón rítmico de batería en presets pentecostales y 4/4
            TASKS["t_pente"] = {"id": "t_pente", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="t_pente",
                style="coro pentecostal",
                lyrics="Fuego pentecostal",
                title="Pentecostal Test",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="random",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            payload_pente = captured_payloads[-1]
            style_pente = payload_pente["style"].lower()
            assert "tempo 4/4" in style_pente
            assert "snare 2-4" in style_pente
            assert "kick 1-3" in style_pente
            assert "hit-hat" in style_pente or "hi-hat" in style_pente
            # Verificación de batería al frente y bien marcada:
            assert "upfront" in style_pente or "al frente" in style_pente or "front of the mix" in style_pente
            # Verificación de bajo eléctrico real y cero synth bass:
            assert "electric bass" in style_pente or "bajo" in style_pente
            assert "zero synth bass" in style_pente
            # Verificación de exclusión de afro-gospel bias:
            assert "no black gospel" in style_pente or "no afro gospel" in style_pente or "no soul" in style_pente

    asyncio.run(run_family_tests())
    print("   -> ÉXITO: Las 3 familias y los presets especiales respetan instrumentos, tempo propio y exclusiones del usuario.")

    # ---------------------------------------------------------------------
    # 7. TEST: Sección de Soporte UDIO para rodryandy y restricción de acceso
    # ---------------------------------------------------------------------
    print("\n[7] Probando endpoint de soporte UDIO y restricción de acceso...")
    
    # 7.1 Acceso anónimo debe ser denegado (401 o 403)
    app.dependency_overrides.clear()
    r_anon = client.get("/api/admin/support_info")
    assert r_anon.status_code in [401, 403], f"Esperado 401/403, obtenido {r_anon.status_code}"
    print("   -> ÉXITO: Acceso anónimo denegado correctamente (HTTP " + str(r_anon.status_code) + ").")

    # 7.2 Acceso con usuario no administrador debe devolver 403
    from fastapi import Request
    
    # Simular usuario regular no admin
    async def mock_regular_user(request: Request):
        return "usuario_comun@gmail.com"
        
    app.dependency_overrides[get_current_user] = mock_regular_user
    r_regular = client.get("/api/admin/support_info")
    assert r_regular.status_code == 403, f"Esperado 403 para usuario común, obtenido {r_regular.status_code}"
    print("   -> ÉXITO: Acceso denegado a usuario no administrador (HTTP 403 Forbidden).")

    # 7.3 Acceso con usuario administrador rodryandy@gmail.com debe devolver 200 con datos de soporte
    async def mock_admin_user(request: Request):
        return "rodryandy@gmail.com"

    app.dependency_overrides[get_current_user] = mock_admin_user
    r_admin = client.get("/api/admin/support_info")
    assert r_admin.status_code == 200, f"Esperado 200 para admin, obtenido {r_admin.status_code}"
    admin_data = r_admin.json()
    assert admin_data.get("authorized") is True
    assert admin_data.get("email") == "support@udioapi.pro"
    assert "Rodrix Music Creator - Solicitud de soporte" in admin_data.get("subject", "")
    assert "identificador de tarea del proveedor" in admin_data.get("body", "")
    print("   -> ÉXITO: Acceso concedido exclusivamente al administrador rodryandy con datos correctos.")

    # Limpiar overrides
    app.dependency_overrides.clear()

    # 7.4 Verificación en index.html
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    assert "modalAdminSupportSection" in html_content
    assert "support@udioapi.pro" in html_content
    assert "contactUdioSupport('gmail')" in html_content
    assert "contactUdioSupport('mailto')" in html_content
    assert "display:none" in html_content  # Inicialmente oculto
    print("   -> ÉXITO: Interfaz index.html incluye la sección 'Soporte UDIO', botones de Gmail y mailto, y restricción por is_admin.")

    # ---------------------------------------------------------------------
    # 8. TEST: Articulación de Batería, Bajo y Selección Épica de Piano de Oración
    # ---------------------------------------------------------------------
    print("\n[8] Probando articulaciones rítmicas detalladas y selección de Piano de Oración...")
    
    # 8.1 Verificaciones en index.html
    assert 'name="hihat_mode"' in html_content
    assert 'value="8th"' in html_content
    assert 'value="4th"' in html_content
    assert 'value="16th"' in html_content
    assert 'id="chkCrashToms"' in html_content
    assert 'id="chkSplashRide"' in html_content
    assert 'id="chkTomGrooves"' in html_content
    assert 'id="chkBassSlap"' in html_content
    assert 'id="chkBassPickup"' in html_content
    assert 'updateHiHatInStylePrompt' in html_content
    # Verificaciones de Piano de Oración (Roland MKS-20, Galaxy DX, Motif XS)
    assert 'name="prayer_piano_type"' in html_content
    assert 'value="mks20"' in html_content
    assert 'value="galaxydx"' in html_content
    assert 'value="motifxs"' in html_content
    assert 'value="acustico"' in html_content
    assert 'PRAYER_PIANO_SOUNDS' in html_content
    assert 'data-gear="roland mks-20 e-piano 2"' in html_content
    assert 'data-gear="galaxy dx"' in html_content
    assert 'data-gear="yamaha motif xs"' in html_content
    print("   -> ÉXITO: Interfaz index.html contiene selectores de Hi-Hat, toms/crashes, bajo slap/pick-up y los 4 tipos de Piano de Oración (MKS-20, Galaxy DX, Motif XS, Acústico).")

    # 8.2 Verificaciones de enrich_style y run_transform_task en app.py
    from app import run_transform_task
    articulation_payloads = []
    async def mock_art_post(client_self, url, json=None, headers=None, **kwargs):
        if json:
            articulation_payloads.append(json)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"workId": "test_mock_work_id"}'
        mock_resp.json.return_value = {"workId": "test_mock_work_id"}
        return mock_resp

    async def mock_art_get(client_self, url, headers=None, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "SUCCESS"}'
        mock_resp.json.return_value = {
            "status": "SUCCESS",
            "data": [{
                "id": "trk_1", "title": "Test", "audio_url": "https://example.com/audio.mp3",
                "status": "SUCCESS", "lyrics": "Test"
            }]
        }
        return mock_resp

    async def run_rhythm_articulation_test():
        with patch("httpx.AsyncClient.post", new=mock_art_post), \
             patch("httpx.AsyncClient.get", new=mock_art_get):
            
            # Caso A: Preset Pentecostal con Hi-Hat en Negras (1 golpe por tempo) y Slap
            TASKS["test_hihat_4th"] = {"id": "test_hihat_4th", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="test_hihat_4th",
                style="coro pentecostal, hi-hat en negras, crashes y redobles, bajo slap",
                title="Test HiHat 4th",
                lyrics="Alabanza...",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            sent_style_a = articulation_payloads[-1]["style"].lower()
            assert "one stroke per beat" in sent_style_a or "quarter-note" in sent_style_a
            assert "slap and pop" in sent_style_a
            assert "crash cymbals" in sent_style_a
            
            # Caso B: Preset Pentecostal con Hi-Hat en Corcheas (2 golpes por tempo) y Pick-up
            TASKS["test_hihat_8th"] = {"id": "test_hihat_8th", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="test_hihat_8th",
                style="coro pentecostal, hi-hat en corcheas, bajo pick-up, grooves en toms",
                title="Test HiHat 8th",
                lyrics="Alabanza...",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=True,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            sent_style_b = articulation_payloads[-1]["style"].lower()
            assert "eighth notes" in sent_style_b or "two strokes per beat" in sent_style_b
            assert "pick-up" in sent_style_b or "passing tone" in sent_style_b
            assert "tom groove" in sent_style_b

            # Caso C: Preset Piano de Oración con Roland MKS-20 Elec Piano 2 (Keyscape)
            TASKS["test_mks20"] = {"id": "test_mks20", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="test_mks20",
                style="piano de oracion, roland mks-20 preset elec piano 2",
                title="Test MKS20",
                lyrics="Oracion...",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=False,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            sent_style_c = articulation_payloads[-1]["style"].lower()
            assert "mks-20" in sent_style_c
            assert "elec piano 2" in sent_style_c or "chorus" in sent_style_c or "keyscape" in sent_style_c

            # Caso D: Preset Piano de Oración con Galaxy DX
            TASKS["test_galaxydx"] = {"id": "test_galaxydx", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="test_galaxydx",
                style="piano de oracion, galaxy dx worship piano",
                title="Test Galaxy DX",
                lyrics="Oracion...",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=False,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            sent_style_d = articulation_payloads[-1]["style"].lower()
            assert "galaxy dx" in sent_style_d
            assert "crystal bells" in sent_style_d or "celestial pad" in sent_style_d

            # Caso E: Preset Piano de Oración con Yamaha Motif XS
            TASKS["test_motifxs"] = {"id": "test_motifxs", "status": "IN_PROGRESS", "logs": ""}
            await run_transform_task(
                task_id="test_motifxs",
                style="piano de oracion, yamaha motif xs",
                title="Test Motif XS",
                lyrics="Oracion...",
                audio_content=None,
                audio_filename=None,
                ignore_audio=True,
                include_lyrics=False,
                bypass_copyright=False,
                exclude_styles="",
                vocal_gender="male",
                weirdness=25,
                style_influence=50,
                audio_influence=25,
                model="chirp-v5",
                pronunciation="",
                obfuscate=False
            )
            sent_style_e = articulation_payloads[-1]["style"].lower()
            assert "motif xs" in sent_style_e
            assert "power grand" in sent_style_e or "choir pad" in sent_style_e or "hall reverb" in sent_style_e
            
    asyncio.run(run_rhythm_articulation_test())
    print("   -> ÉXITO: Los estilos y tareas de transformación enriquecen e interpretan con fidelidad las subdivisiones de Hi-Hat, adornos slap/pick-up y los 4 pianos de oración (MKS-20 Keyscape, Galaxy DX, Motif XS, Acústico).")

    print("\n=== TODAS LAS VERIFICACIONES FINALES RESULTARON EXITOSAS (8/8) ===")

if __name__ == "__main__":
    test_suite()
