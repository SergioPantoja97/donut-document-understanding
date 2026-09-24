import json
import re
import time

import streamlit as st
import torch
from PIL import Image
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
        with st.spinner("Procesando documento..."):
            output = donut_inference(
                image=image,
                processor=processor,
                model=model,
                device=device,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

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

        with st.expander("Ver secuencia limpia generada por Donut"):
            st.code(output["clean_sequence"], language="text")

        with st.expander("Ver secuencia raw generada por Donut"):
            st.code(output["raw_sequence"], language="text")

        if not output["parse_ok"]:
            st.warning("La salida no se pudo convertir perfectamente a JSON.")
            st.write("Detalle:", output["parse_error"])

