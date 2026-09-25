import json
import re
import time
import numpy as np

import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Donut - Document Understanding",
    page_icon="🍩",
    layout="wide"
)

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"

# ============================================================
# FUNCIONES
# ============================================================
@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return processor, model, device


def donut_inference(
    image: Image.Image,
    processor,
    model,
    device,
    max_length=512,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
):
    image = image.convert("RGB")

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(device)

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            do_sample=False,
            num_beams=1,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    inference_time = time.perf_counter() - start_time

    raw_sequence = processor.batch_decode(outputs)[0]

    sequence = raw_sequence.replace(
        processor.tokenizer.eos_token or "",
        ""
    ).replace(
        processor.tokenizer.pad_token or "",
        ""
    )

    # Eliminar solo el primer token de tarea, por ejemplo <s_cord-v2>
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    try:
        result = processor.token2json(sequence)
        parse_ok = True
        parse_error = None
    except Exception as e:
        result = {"raw_output": sequence}
        parse_ok = False
        parse_error = str(e)

    return {
        "raw_sequence": raw_sequence,
        "clean_sequence": sequence,
        "result": result,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "inference_time": inference_time,
        "tokens_generated": int(outputs.shape[-1]),
    }




def detect_visual_regions(image: Image.Image, cols=8, rows=12, top_k=14):
    """
    Heurística visual para localizar regiones con alta variación/contraste.
    Sirve SOLO para la simulación didáctica del flujo.
    No es OCR y no representa atención real del modelo.
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    h, w = arr.shape
    regions = []

    for r in range(rows):
        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)
            y0 = int(r * h / rows)
            y1 = int((r + 1) * h / rows)

            patch = arr[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            # Mezcla de contraste y oscuridad para priorizar zonas tipo texto.
            contrast = float(np.std(patch))
            darkness = float(255.0 - np.mean(patch))
            score = contrast * 0.75 + darkness * 0.25

            regions.append((score, (x0, y0, x1, y1)))

    regions.sort(key=lambda x: x[0], reverse=True)
    return [box for _, box in regions[:top_k]]


def draw_regions_frame(image: Image.Image, regions, visible_count, active_index=None):
    frame = image.copy().convert("RGB")
    draw = ImageDraw.Draw(frame, "RGBA")

    for i, (x0, y0, x1, y1) in enumerate(regions[:visible_count]):
        if active_index is not None and i == active_index:
            fill = (60, 220, 100, 70)
            outline = (20, 170, 70, 255)
            width = 5
        else:
            fill = (70, 210, 110, 38)
            outline = (30, 180, 80, 210)
            width = 3

        draw.rectangle(
            [x0, y0, x1, y1],
            fill=fill,
            outline=outline,
            width=width
        )

    return frame


def animate_visual_simulation(image: Image.Image, placeholder, status_placeholder):
    """
    Simulación visual progresiva del recorrido:
    imagen -> regiones visuales -> encoder -> representación.

    IMPORTANTE:
    no son bounding boxes OCR ni mapas reales de atención.
    """
    regions = detect_visual_regions(image)

    status_placeholder.info("Simulación visual: analizando regiones del documento...")

    # Mostrar cajas progresivamente.
    for i in range(1, len(regions) + 1):
        frame = draw_regions_frame(
            image,
            regions,
            visible_count=i,
            active_index=i - 1
        )

        placeholder.image(
            frame,
            caption=f"Simulación visual — región {i} de {len(regions)}",
            use_container_width=True
        )
        time.sleep(0.10)

    status_placeholder.info("Simulación visual: encoder construyendo representaciones...")

    # Pulso visual sobre todas las zonas seleccionadas.
    for _ in range(2):
        frame = image.copy().convert("RGB")
        draw = ImageDraw.Draw(frame, "RGBA")

        for x0, y0, x1, y1 in regions:
            draw.rectangle(
                [x0, y0, x1, y1],
                fill=(40, 210, 90, 55),
                outline=(20, 170, 70, 255),
                width=4
            )

        placeholder.image(
            frame,
            caption="Regiones visuales destacadas para la simulación del encoder",
            use_container_width=True
        )
        time.sleep(0.25)

    return regions


def truncate_text(value, max_chars=1200):
    value = str(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n\n... [salida recortada]"


def show_pipeline_simulation(image, output, regions):
    st.markdown("### Simulación visual del procesamiento")

    st.warning(
        "Visualización didáctica: las cajas verdes son regiones detectadas por una heurística "
        "de contraste para explicar el flujo. No son OCR ni pesos reales de atención."
    )

    c1, c2, c3 = st.columns([1.15, 1, 1])

    with c1:
        st.markdown("#### 1. Regiones visuales")
        final_frame = draw_regions_frame(
            image,
            regions,
            visible_count=len(regions)
        )
        st.image(
            final_frame,
            caption="Regiones visuales resaltadas",
            use_container_width=True
        )
        st.caption("Simulación de zonas relevantes antes de la representación interna.")

    with c2:
        st.markdown("#### 2. Secuencia generada")
        st.code(
            truncate_text(output["clean_sequence"], 1000),
            language="text"
        )
        st.caption("El decoder genera tokens de forma autoregresiva.")

    with c3:
        st.markdown("#### 3. Resultado estructurado")
        st.json(output["result"], expanded=False)
        st.caption("La secuencia se transforma a una estructura tipo JSON.")

    st.markdown("#### Q, K, V y Cross-Attention")

    q, k, v = st.columns(3)
    with q:
        st.markdown("**Q · Query**")
        st.write("Proviene del decoder: qué información necesita para generar el siguiente token.")
    with k:
        st.markdown("**K · Key**")
        st.write("Proviene del encoder: ayuda a identificar qué representaciones visuales son relevantes.")
    with v:
        st.markdown("**V · Value**")
        st.write("Proviene del encoder: contiene la información visual que se incorpora al contexto.")

    st.info(
        "En cross-attention, el decoder compara Q con K y usa los V asociados "
        "para producir el siguiente token."
    )


# ============================================================
# INTERFAZ
# ============================================================
st.title("🍩 Donut - Document Understanding Transformer")
st.markdown(
    """
Esta aplicación permite cargar una imagen de un documento o recibo y ejecutar inferencia
con **Donut** para obtener una salida estructurada.
"""
)

with st.sidebar:
    st.header("Configuración")
    max_length = st.slider("max_length", min_value=128, max_value=1024, value=512, step=64)
    repetition_penalty = st.slider("repetition_penalty", min_value=1.00, max_value=2.00, value=1.15, step=0.05)
    no_repeat_ngram_size = st.slider("no_repeat_ngram_size", min_value=1, max_value=8, value=4, step=1)


processor, model, device = load_model()

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("1. Cargar documento")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Documento cargado", use_container_width=True)
    else:
        image = None
        st.info("Carga una imagen para iniciar la inferencia.")

with col_b:
    st.subheader("2. Ejecutar inferencia")
    run_button = st.button("Procesar documento", type="primary", use_container_width=True)

if run_button:
    if image is None:
        st.warning("Primero debes cargar una imagen.")
    else:
        st.markdown("### Procesamiento en tiempo real")
        live_col1, live_col2 = st.columns([1.2, 1])

        with live_col1:
            live_image = st.empty()

        with live_col2:
            live_status = st.empty()
            live_status.info("Iniciando simulación visual...")

        regions = animate_visual_simulation(
            image=image,
            placeholder=live_image,
            status_placeholder=live_status
        )

        live_status.info("Ejecutando inferencia real con Donut...")

        with st.spinner("Procesando documento con el modelo..."):
            output = donut_inference(
                image=image,
                processor=processor,
                model=model,
                device=device,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        live_status.success("Inferencia completada.")
        st.success("Inferencia completada.")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Tiempo de inferencia", f"{output['inference_time']:.2f} s")
        metric2.metric("Tokens generados", output["tokens_generated"])
        metric3.metric("Dispositivo", str(device).upper())

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Documento")
            st.image(image, caption="Imagen analizada", use_container_width=True)

        with right:
            st.subheader("Resultado estructurado")
            st.json(output["result"], expanded=True)

            json_str = json.dumps(output["result"], indent=2, ensure_ascii=False)
            st.download_button(
                label="Descargar JSON",
                data=json_str,
                file_name="resultado_donut.json",
                mime="application/json",
                use_container_width=True
            )

        show_pipeline_simulation(image, output, regions)

        with st.expander("Ver secuencia limpia generada por Donut"):
            st.code(output["clean_sequence"], language="text")

        with st.expander("Ver secuencia raw generada por Donut"):
            st.code(output["raw_sequence"], language="text")

        if not output["parse_ok"]:
            st.warning("La salida no se pudo convertir perfectamente a JSON.")
            st.write("Detalle:", output["parse_error"])