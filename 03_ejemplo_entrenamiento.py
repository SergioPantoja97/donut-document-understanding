"""
03_ejemplo_entrenamiento.py
===========================

Ejemplo didáctico de FINE-TUNING de Donut sobre un dominio propio (facturas).

Responde la pregunta: "¿Cómo se entrena Donut si no hay texto OCR?"
Respuesta: NO se le da texto OCR. Se le da la IMAGEN y la SALIDA ESTRUCTURADA
(un JSON/secuencia objetivo). El modelo aprende a mapear imagen -> secuencia
directamente. El objetivo de entrenamiento es la cross-entropy sobre los
tokens de la secuencia objetivo (teacher forcing).

Este script:
  1. Genera facturas sintéticas (imagen + JSON objetivo) con PIL.
  2. Arma un torch Dataset que preprocesa imagen y tokeniza la secuencia.
  3. Muestra un mini bucle de entrenamiento (unas pocas iteraciones) para
     ilustrar el mecanismo. NO busca converger.

Ejecución:
    python 03_ejemplo_entrenamiento.py
"""

import json
import random

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_NAME = "naver-clova-ix/donut-base"


# ---------------------------------------------------------------------------
# 1. Generación de facturas sintéticas
# ---------------------------------------------------------------------------
def _fuente(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def generar_factura(seed: int):
    """Devuelve (imagen PIL, dict de campos objetivo)."""
    random.seed(seed)
    tienda = random.choice(["Cafe Central", "Super Uno", "Libreria Sol", "Farma Plus"])
    total = round(random.uniform(5, 500), 2)
    fecha = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    n_items = random.randint(1, 3)

    img = Image.new("RGB", (400, 500), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), tienda, fill="black", font=_fuente(24))
    draw.text((20, 60), f"Fecha: {fecha}", fill="black", font=_fuente(16))
    draw.line([(20, 90), (380, 90)], fill="black", width=1)

    y = 110
    items = []
    for _ in range(n_items):
        nombre = random.choice(["Item A", "Item B", "Item C", "Item D"])
        precio = round(random.uniform(1, 100), 2)
        items.append({"nombre": nombre, "precio": f"{precio:.2f}"})
        draw.text((20, y), nombre, fill="black", font=_fuente(16))
        draw.text((280, y), f"{precio:.2f}", fill="black", font=_fuente(16))
        y += 30

    draw.line([(20, y + 10), (380, y + 10)], fill="black", width=1)
    draw.text((20, y + 25), "TOTAL", fill="black", font=_fuente(18))
    draw.text((280, y + 25), f"{total:.2f}", fill="black", font=_fuente(18))

    objetivo = {"tienda": tienda, "fecha": fecha, "total": f"{total:.2f}", "items": items}
    return img, objetivo


def json_a_secuencia(processor, d: dict) -> str:
    """Convierte un dict a la secuencia de tokens especiales estilo Donut."""
    # transformers ofrece json2token en el DonutProcessor
    seq = processor.token2json  # noqa: F841  (referencia inversa disponible)
    return (
        processor.tokenizer.bos_token
        + processor.json2token(d, sort_json_key=False)
        + processor.tokenizer.eos_token
    )


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class FacturasDataset(Dataset):
    def __init__(self, processor, n: int = 8, max_length: int = 128):
        self.processor = processor
        self.max_length = max_length
        self.muestras = [generar_factura(i) for i in range(n)]

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        imagen, objetivo = self.muestras[idx]
        pixel_values = self.processor(imagen, return_tensors="pt").pixel_values.squeeze(0)

        secuencia = json_a_secuencia(self.processor, objetivo)
        labels = self.processor.tokenizer(
            secuencia,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Ignorar padding en la pérdida
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# 3. Mini bucle de entrenamiento (ilustrativo)
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

    # Añadir tokens especiales nuevos que aparecen en las secuencias objetivo
    ejemplo_img, ejemplo_obj = generar_factura(0)
    _ = json_a_secuencia(processor, ejemplo_obj)  # fuerza registro de tokens
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Config de generación / tokens de arranque
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    dataset = FacturasDataset(processor, n=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    print("\n--- Mini bucle de entrenamiento (2 epochs, ilustrativo) ---")
    print("La pérdida = cross-entropy entre los tokens generados y la secuencia objetivo.")
    print("Nótese: NUNCA se pasa texto OCR; solo imagen + secuencia objetivo.\n")

    for epoch in range(2):
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch {epoch}  step {step}  loss = {loss.item():.4f}")

    print("\n[OK] Demostración de entrenamiento completada.")
    print("Para un fine-tuning real: más datos, más epochs, scheduler y validación.")

    # Guardar un ejemplo de par (imagen, objetivo) para inspección
    ejemplo_img.save("factura_ejemplo.png")
    with open("factura_ejemplo.json", "w", encoding="utf-8") as f:
        json.dump(ejemplo_obj, f, ensure_ascii=False, indent=2)
    print("Guardados: factura_ejemplo.png y factura_ejemplo.json")


if __name__ == "__main__":
    main()
