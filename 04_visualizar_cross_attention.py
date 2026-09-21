"""
04_visualizar_cross_attention.py
================================

VALOR AGREGADO: visualiza QUÉ "mira" Donut en la imagen mediante el
cross-attention del decoder.

Idea:
  - Corremos la inferencia pidiendo output_attentions=True.
  - Para el (o los) token(s) generados, tomamos los pesos de cross-attention
    (cuánto atiende cada token de salida a cada patch del encoder).
  - Reproyectamos esos pesos a la cuadrícula espacial de patches y los
    superponemos como un mapa de calor sobre la imagen original.

Salidas:
  - cross_attention_overlay.png  (imagen + heatmap superpuesto)
  - cross_attention_grid.png     (imagen | heatmap | overlay)

Nota: requiere transformers, torch, matplotlib, numpy, Pillow. En CPU es lento
pero funciona. En Colab con GPU es inmediato.

Ejecución:
    python3 04_visualizar_cross_attention.py
"""

import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"


def crear_recibo():
    img = Image.new("RGB", (480, 560), color="white")
    d = ImageDraw.Draw(img)

    def fnt(s):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", s)
        except OSError:
            return ImageFont.load_default()

    d.text((20, 20), "CAFE CENTRAL", fill="black", font=fnt(26))
    d.text((20, 60), "Fecha: 2024-05-14", fill="black", font=fnt(16))
    d.line([(20, 92), (460, 92)], fill="black", width=1)
    y = 110
    for nombre, precio in [("Cafe Americano", "3.50"),
                           ("Croissant", "2.75"),
                           ("Jugo Naranja", "4.20")]:
        d.text((20, y), nombre, fill="black", font=fnt(18))
        d.text((360, y), precio, fill="black", font=fnt(18))
        y += 34
    d.line([(20, y + 8), (460, y + 8)], fill="black", width=1)
    d.text((20, y + 24), "TOTAL", fill="black", font=fnt(20))
    d.text((360, y + 24), "10.45", fill="black", font=fnt(20))
    return img


@torch.no_grad()
def visualizar(processor, model, image, device, out_prefix="cross_attention"):
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    gen = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    # gen.cross_attentions: tupla por paso de generación; cada elemento es
    # una tupla por capa del decoder con forma (batch, heads, q_len, kv_len).
    # Promediamos sobre pasos, capas y cabezas para obtener un mapa global.
    pasos = gen.cross_attentions
    acumulado = None
    n = 0
    for paso in pasos:
        for capa in paso:
            # capa: (batch, heads, q_len, kv_len) -> tomamos el último token query
            a = capa[0].mean(0)          # promedio sobre heads -> (q_len, kv_len)
            a = a[-1]                    # último token generado -> (kv_len,)
            acumulado = a if acumulado is None else acumulado + a
            n += 1
    mapa = (acumulado / max(n, 1)).float().cpu().numpy()

    # Reproyectar el vector (kv_len,) a una cuadrícula 2D de patches.
    kv = mapa.shape[0]
    lado = int(round(math.sqrt(kv)))
    # Ajuste: recortar/rellenar para poder hacer reshape cuadrado aproximado.
    lado = max(lado, 1)
    usable = lado * lado
    grid = mapa[:usable].reshape(lado, lado)
    grid = (grid - grid.min()) / (grid.ptp() + 1e-8)

    # Redimensionar el heatmap al tamaño de la imagen
    heat = Image.fromarray((grid * 255).astype("uint8")).resize(image.size, Image.BILINEAR)
    heat = np.asarray(heat) / 255.0

    # Figura combinada: imagen | heatmap | overlay
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].imshow(image); ax[0].set_title("1) Imagen de entrada"); ax[0].axis("off")
    ax[1].imshow(heat, cmap="jet"); ax[1].set_title("2) Cross-attention (dónde mira)"); ax[1].axis("off")
    ax[2].imshow(image); ax[2].imshow(heat, cmap="jet", alpha=0.45)
    ax[2].set_title("3) Overlay: atención sobre la imagen"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_grid.png", dpi=130, bbox_inches="tight")
    print(f"[OK] Guardado: {out_prefix}_grid.png")

    # Overlay solo
    plt.figure(figsize=(6, 7))
    plt.imshow(image); plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.title("Cross-attention: qué mira Donut al generar la salida")
    plt.axis("off")
    plt.savefig(f"{out_prefix}_overlay.png", dpi=130, bbox_inches="tight")
    print(f"[OK] Guardado: {out_prefix}_overlay.png")

    seq = processor.batch_decode(gen.sequences)[0]
    return seq


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    image = crear_recibo()
    seq = visualizar(processor, model, image, device)
    print("\nSalida del modelo (cruda):")
    print(seq)


if __name__ == "__main__":
    main()
