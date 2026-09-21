"""
01_inferencia_matriculas.py
============================

Demo de inferencia con Donut (naver-clova-ix/donut-base) sobre una imagen
sintética de un documento (estilo matrícula / registro vehicular).

Muestra el flujo completo OCR-free:

    Imagen  ->  DonutProcessor  ->  VisionEncoderDecoderModel.generate()  ->  Texto/JSON

No requiere datos externos: la imagen se genera con PIL dentro del script.

Ejecución:
    python 01_inferencia_matriculas.py
"""

import re

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Modelo fine-tuneado sobre CORD (parsing de recibos): produce JSON real.
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"


# ---------------------------------------------------------------------------
# 1. Crear una imagen sintética de documento (matrícula dominicana ficticia)
# ---------------------------------------------------------------------------
def crear_documento_sintetico(path: str = "documento_matricula.png") -> Image.Image:
    """Genera una imagen sintética que simula una matrícula vehicular."""
    ancho, alto = 640, 400
    img = Image.new("RGB", (ancho, alto), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    def fuente(size: int):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()

    # Encabezado
    draw.rectangle([0, 0, ancho, 60], fill=(0, 70, 140))
    draw.text((20, 18), "REPUBLICA DOMINICANA - DGII", fill="white", font=fuente(22))

    # Cuerpo del documento (pares campo: valor)
    campos = [
        ("MATRICULA:", "A-1234567"),
        ("PLACA:", "K012345"),
        ("MARCA:", "TOYOTA"),
        ("MODELO:", "COROLLA"),
        ("ANO:", "2021"),
        ("COLOR:", "GRIS"),
        ("PROPIETARIO:", "JUAN PEREZ"),
    ]

    y = 90
    for etiqueta, valor in campos:
        draw.text((30, y), etiqueta, fill=(30, 30, 30), font=fuente(20))
        draw.text((260, y), valor, fill=(0, 0, 0), font=fuente(20))
        y += 42

    img.save(path)
    print(f"[OK] Imagen sintética guardada en: {path}")
    return img


# ---------------------------------------------------------------------------
# 2. Cargar modelo y processor
# ---------------------------------------------------------------------------
def cargar_modelo(device: str):
    print(f"[..] Cargando '{MODEL_NAME}' (puede tardar la primera vez)...")
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Modelo cargado. Parámetros: {n_params/1e6:.1f}M")
    return processor, model


# ---------------------------------------------------------------------------
# 3. Inferencia OCR-free
# ---------------------------------------------------------------------------
@torch.no_grad()
def inferencia(processor, model, imagen: Image.Image, device: str) -> str:
    # El prompt de tarea condiciona al decoder. donut-base usa un prompt base.
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    pixel_values = processor(imagen, return_tensors="pt").pixel_values.to(device)

    print("[..] Ejecutando model.generate() (decoder autoregresivo)...")
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    secuencia = processor.batch_decode(outputs.sequences)[0]
    # Limpieza de tokens especiales
    secuencia = secuencia.replace(processor.tokenizer.eos_token, "")
    secuencia = secuencia.replace(processor.tokenizer.pad_token, "")
    secuencia = re.sub(r"<.*?>", "", secuencia, count=1).strip()
    return secuencia


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    imagen = crear_documento_sintetico()
    processor, model = cargar_modelo(device)
    resultado = inferencia(processor, model, imagen, device)

    print("\n" + "=" * 60)
    print("RESULTADO DE LA INFERENCIA (salida cruda del decoder):")
    print("=" * 60)
    print(resultado)
    print("=" * 60)
    print(
        "\nNota: donut-base es el checkpoint pre-entrenado (sin fine-tune para\n"
        "este tipo de documento), por lo que la salida es ilustrativa del\n"
        "mecanismo, no un parseo perfecto. Para resultados reales sobre un\n"
        "dominio se hace fine-tuning (ver 03_ejemplo_entrenamiento.py)."
    )


if __name__ == "__main__":
    main()
