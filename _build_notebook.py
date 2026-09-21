"""
Genera Donut_Colab_Notebook.ipynb usando la librería estándar json
(JSON de notebook válido garantizado, sin dependencias externas).
Uso: python3 _build_notebook.py
"""
import json

cells = []


def _split_lines(text):
    """Divide el texto en 'source' formato notebook: lista de líneas con \n final,
    salvo la última (convención de nbformat)."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_lines(text),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_lines(text),
    })


# ------------------------------------------------------------------ Sección 1
md(r"""# 🍩 Donut — OCR-free Document Understanding Transformer

> Notebook educativo · ECCV 2022 · [arXiv:2111.15664](https://arxiv.org/abs/2111.15664)

**Enlaces:**
- 📄 Paper: https://arxiv.org/abs/2111.15664
- 💻 Código oficial: https://github.com/clovaai/donut
- 🤗 Modelo: https://huggingface.co/naver-clova-ix/donut-base

**¿Qué es Donut?**
Donut (Document Understanding Transformer) lee imágenes de documentos y produce
texto/JSON estructurado **sin usar un motor OCR externo**. La imagen entra
directo al modelo: un **encoder de visión (Swin Transformer)** convierte la
imagen en vectores y un **decoder de texto (Transformer)** genera la respuesta
token por token, mirando esos vectores mediante **cross-attention**.

Este notebook explica la arquitectura y ejecuta una inferencia completa en Colab
con GPU gratuita. No requiere datos externos: la imagen de prueba se crea aquí
mismo con PIL.
""")

# ------------------------------------------------------------------ Sección 2
md(r"""## 2. Instalación e imports

Ejecuta esta celda primero. En Colab activa la GPU:
`Entorno de ejecución → Cambiar tipo de entorno → GPU`.

Instalamos `transformers` desde el repositorio oficial de Hugging Face (para
tener la última implementación de Donut/VisionEncoderDecoder), `accelerate` y
`timm` (el encoder Swin depende de `timm`), y **`sentencepiece` + `protobuf`**,
que el tokenizer de Donut (XLM-RoBERTa) necesita. Sin `protobuf` verás el error
`XLMRobertaConverter requires the protobuf library`.

⚠️ **IMPORTANTE:** tras instalar, **reinicia el entorno de ejecución**
(`Entorno de ejecución → Reiniciar entorno de ejecución`) y luego ejecuta desde
la celda de imports en adelante, SIN volver a correr esta celda. Esto es
necesario porque `protobuf`/`sentencepiece` se cargan al iniciar el proceso.""")

code(r"""# Instalación de dependencias — incluye sentencepiece y protobuf (obligatorios)
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q accelerate timm sentencepiece "protobuf<4" Pillow matplotlib numpy

# En Colab, reinicia el runtime automáticamente para que protobuf/sentencepiece
# queden bien cargados. Tras el reinicio, salta esta celda y sigue desde imports.
import os
try:
    import google.protobuf  # noqa: F401
    import sentencepiece    # noqa: F401
    print("protobuf y sentencepiece OK — puedes continuar con la siguiente celda.")
except ImportError:
    print("Reiniciando runtime para cargar protobuf/sentencepiece...")
    os.kill(os.getpid(), 9)""")

code(r"""import re
import json
import textwrap
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import DonutProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch:", torch.__version__)
print("¿GPU disponible?:", torch.cuda.is_available())
print("Dispositivo:", device)""")

# ------------------------------------------------------------------ Sección 3
md(r"""## 3. ¿Qué es Donut? Pipeline tradicional vs Donut

En un pipeline tradicional de comprensión de documentos hay un paso de OCR
separado que reconoce el texto, y luego un modelo de NLP lo interpreta. Ese OCR
introduce errores que se propagan y encarece el sistema.

Donut elimina ese paso: la imagen va **directo** al modelo.

```
PIPELINE TRADICIONAL
  Imagen ──▶ [ OCR externo ] ──▶ Texto ──▶ [ Modelo NLP ] ──▶ Salida
                  ▲ errores de OCR se propagan

PIPELINE DONUT (OCR-free)
  Imagen ──▶ [ Encoder Swin ] ──▶ vectores ──▶ [ Decoder + cross-attn ] ──▶ JSON/Texto
                  end-to-end, entrenado de punta a punta
```
""")

# ------------------------------------------------------------------ Sección 4
md(r"""## 4. Arquitectura general (sección clave)

### 4.1 Visión general del flujo

```
Imagen → [Encoder: Swin Transformer] → [Cross-Attention] → [Decoder: Transformer] → JSON/Texto
```

Donut es un modelo **encoder–decoder** (arquitectura *VisionEncoderDecoder*).

### 4.2 Encoder — Swin Transformer
- **Qué hace:** convierte la imagen en una secuencia de vectores.
- **Cómo:** divide la imagen en *patches*, los proyecta linealmente a embeddings
  y aplica capas de atención con **ventanas deslizantes** (*shifted windows*),
  que capturan contexto local y global de forma eficiente.
- **Tipo:** similar a un ViT, pero jerárquico y con ventanas desplazadas.
- **Salida:** una secuencia de `N` vectores de dimensión `D` → forma `(N, D)`.

### 4.3 Decoder — Transformer decoder estándar
- **Qué hace:** genera texto **token por token** de forma autoregresiva.
- **Arquitectura:** capas de *self-attention* + *cross-attention* + *feed-forward*.
- Usa **positional encoding** para saber la posición de cada token.
- Termina cuando emite el token de fin `<eos>`.

### 4.4 Cross-Attention (el punto clave) 🔑
- El decoder tiene una capa de **cross-attention** que atiende a la salida del
  encoder (los vectores de la imagen, la "memoria").
- Así el decoder **"mira" la imagen** al generar cada token.
- Es la **conexión** entre encoder y decoder: el encoder produce los vectores de
  la imagen y el decoder los usa como memoria externa.

### 4.5 Flujo completo paso a paso
1. **Imagen de entrada** (ej. `1280×960×3`, redimensionada por el processor).
2. **Encoder:** imagen → patches → embeddings → Swin → secuencia `(N, D)`.
3. **Decoder input:** token de inicio + positional encoding.
4. **Decoder:** *self-attention* sobre tokens previos + *cross-attention* sobre
   los vectores del encoder → predice el siguiente token.
5. **Repetir** hasta el token de fin `<eos>`.
6. **Salida:** secuencia de tokens que se decodifica a texto/JSON.
""")

# ------------------------------------------------------------------ Sección 5
md(r"""## 5. Carga del modelo pre-entrenado

Usamos `naver-clova-ix/donut-base-finetuned-cord-v2`: el modelo Donut ya
**fine-tuneado sobre CORD** (parsing de recibos). Al estar fine-tuneado para
esa tarea, produce **JSON estructurado real** a partir de la imagen, no solo
una salida ilustrativa.

- `DonutProcessor`: preprocesa la imagen (resize + normalize) y tokeniza.
- `VisionEncoderDecoderModel`: encoder Swin Transformer + decoder Transformer.""")

code(r"""MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

# Si aparece un error de 'protobuf', primero ejecuta la celda de instalación y
# REINICIA el runtime. Como red de seguridad extra, intentamos el tokenizer
# rápido y, si falla por protobuf, caemos al lento (use_fast=False).
try:
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
except ImportError as e:
    print("Tokenizer rápido falló (", e, "). Usando use_fast=False...")
    processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=False)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

enc = model.config.encoder
dec = model.config.decoder
print("Encoder:", enc.model_type, "(Swin Transformer)")
print("Decoder:", dec.model_type, "(Transformer decoder)")
print("Tamaño de imagen esperado:", enc.image_size)
print("Dim. oculta decoder:", dec.hidden_size)
print("Vocabulario decoder:", dec.vocab_size)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")""")

# ------------------------------------------------------------------ Sección 6
md(r"""## 6. Crear una imagen sintética (recibo)

No usamos datos externos: generamos un recibo con PIL y lo mostramos.""")

code(r"""def crear_recibo():
    img = Image.new("RGB", (480, 560), color="white")
    draw = ImageDraw.Draw(img)
    def fnt(s):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", s)
        except OSError:
            return ImageFont.load_default()

    draw.text((20, 20), "CAFE CENTRAL", fill="black", font=fnt(26))
    draw.text((20, 60), "Fecha: 2024-05-14", fill="black", font=fnt(16))
    draw.line([(20, 92), (460, 92)], fill="black", width=1)

    items = [("Cafe Americano", "3.50"), ("Croissant", "2.75"), ("Jugo Naranja", "4.20")]
    y = 110
    for nombre, precio in items:
        draw.text((20, y), nombre, fill="black", font=fnt(18))
        draw.text((360, y), precio, fill="black", font=fnt(18))
        y += 34

    draw.line([(20, y + 8), (460, y + 8)], fill="black", width=1)
    draw.text((20, y + 24), "TOTAL", fill="black", font=fnt(20))
    draw.text((360, y + 24), "10.45", fill="black", font=fnt(20))
    return img

recibo = crear_recibo()
plt.figure(figsize=(4, 5))
plt.imshow(recibo)
plt.axis("off")
plt.title("Imagen sintética de entrada")
plt.show()""")

# ------------------------------------------------------------------ Sección 7
md(r"""## 7. Inferencia completa (OCR-free)

Pasos: preprocesar la imagen → construir el prompt de tarea `<s_cord-v2>` →
`model.generate()` (decoder autoregresivo con cross-attention) → decodificar
y parsear la salida a JSON.

**Elige la imagen:** por defecto usamos el recibo sintético de la sección 6.
Si quieres probar con una factura/recibo real, sube el archivo a Colab
(panel izquierdo → 📁 Archivos) y cambia `IMAGE_PATH` abajo.""")

code(r"""import os

# Imagen a analizar. Por defecto usa 'image.png' (la factura del workspace).
# - En Colab: sube tu foto y pon su ruta, p.ej. "/content/image.png".
# - En local: usa la ruta al archivo, p.ej. "image.png" o la ruta completa.
# Si no se encuentra el archivo, se usa el recibo sintético de la sección 6.
IMAGE_PATH = "image.png"

if IMAGE_PATH and os.path.exists(IMAGE_PATH):
    image = Image.open(IMAGE_PATH).convert("RGB")
    print("Usando imagen:", IMAGE_PATH)
else:
    print("No se encontró", IMAGE_PATH, "-> uso el recibo sintético de la sección 6.")
    image = recibo  # sintético de la sección 6

# Prompt de tarea que condiciona al decoder (tarea CORD v2)
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids.to(device)

# Preprocesar imagen -> pixel_values
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
print("pixel_values.shape =", tuple(pixel_values.shape), "(batch, canales, alto, ancho)")

# Generar (decoder autoregresivo con cross-attention)
outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512,
)

# Decodificar
result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Salida del decoder (texto) ---")
print(result)

# Convertir a JSON estructurado con el helper de Donut
seq = processor.batch_decode(outputs)[0]
seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
seq = re.sub(r"<.*?>", "", seq, count=1).strip()
try:
    estructura = processor.token2json(seq)
    print("\n--- JSON estructurado (token2json) ---")
    print(json.dumps(estructura, indent=2, ensure_ascii=False))
except Exception as e:
    print("No se pudo parsear a JSON:", e)""")

md(r"""### 7.1 Visualización: imagen de entrada + resultado estructurado

Mostramos lado a lado el documento y la salida del modelo.""")

code(r"""fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].imshow(image)
ax[0].axis("off")
ax[0].set_title("Documento de entrada")

# Intentar formatear como JSON; si no, mostrar texto plano
try:
    formatted = json.dumps(json.loads(result), indent=4, ensure_ascii=False)
except Exception:
    try:
        formatted = json.dumps(processor.token2json(seq), indent=4, ensure_ascii=False)
    except Exception:
        formatted = result

wrapped = "\n".join(textwrap.wrap(formatted, width=60))
ax[1].text(0, 0.5, wrapped, fontsize=10, va="center", ha="left", family="monospace")
ax[1].axis("off")
ax[1].set_title("Resultado estructurado")

plt.tight_layout()
plt.show()""")

md(r"""### 7.2 Inspección de tensores y cross-attention

Veamos las formas internas: la "memoria" del encoder y el tensor de
cross-attention que conecta cada token de salida con cada patch de la imagen.""")

code(r"""with torch.no_grad():
    enc_out = model.encoder(pixel_values)
memoria = enc_out.last_hidden_state
print("Memoria del encoder (vectores de imagen):", tuple(memoria.shape), "→ (batch, N_patches, D)")

with torch.no_grad():
    step = model(pixel_values=pixel_values,
                 decoder_input_ids=decoder_input_ids,
                 output_attentions=True)
print("logits:", tuple(step.logits.shape), "→ (batch, seq_len, vocab)")
if step.cross_attentions is not None:
    ca = step.cross_attentions[0]
    print("cross_attention[capa 0]:", tuple(ca.shape),
          "→ (batch, heads, tokens_decoder, N_patches)")
    print("Cada valor: cuánto atiende un token de salida a un patch de la imagen. ¡La conexión!")""")

md(r"""### 7.3 Visualización del cross-attention (valor agregado) 🔥

Superponemos un mapa de calor sobre la imagen para ver **qué mira el modelo**
al generar la salida. Promediamos los pesos de cross-attention sobre pasos,
capas y cabezas, y los reproyectamos a la cuadrícula de patches del encoder.""")

code(r"""import math

# Generamos de nuevo pidiendo las atenciones de todos los pasos
with torch.no_grad():
    gen = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        output_attentions=True,
        return_dict_in_generate=True,
    )

# Promediar cross-attention sobre pasos, capas y cabezas -> vector (N_patches,)
acumulado, n = None, 0
for paso in gen.cross_attentions:
    for capa in paso:
        a = capa[0].mean(0)[-1]  # promedio heads, último token query -> (kv_len,)
        acumulado = a if acumulado is None else acumulado + a
        n += 1
mapa = (acumulado / max(n, 1)).float().cpu().numpy()

# Reproyectar a cuadrícula 2D aproximadamente cuadrada
lado = max(int(round(math.sqrt(mapa.shape[0]))), 1)
grid = mapa[:lado*lado].reshape(lado, lado)
grid = (grid - grid.min()) / (np.ptp(grid) + 1e-8)

heat = np.asarray(
    Image.fromarray((grid*255).astype("uint8")).resize(image.size, Image.BILINEAR)
) / 255.0

fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax[0].imshow(image); ax[0].set_title("1) Imagen"); ax[0].axis("off")
ax[1].imshow(heat, cmap="jet"); ax[1].set_title("2) Cross-attention"); ax[1].axis("off")
ax[2].imshow(image); ax[2].imshow(heat, cmap="jet", alpha=0.45)
ax[2].set_title("3) Overlay (dónde mira)"); ax[2].axis("off")
plt.tight_layout(); plt.show()""")

md(r"""Las zonas cálidas (rojo/amarillo) indican los patches de la imagen a los
que el decoder presta más atención al generar la salida. Es la evidencia visual
de que el cross-attention **conecta el texto generado con la región relevante de
la imagen**.""")

# ------------------------------------------------------------------ Sección 8
md(r"""## 8. Métricas del paper

Resultados reportados por Donut en distintas tareas (valores aproximados; el
exacto depende del setup y checkpoint):

| Tarea | Dataset | Métrica | Donut |
|-------|---------|---------|-------|
| Document VQA | DocVQA | ANLS | ~67.5 |
| Parsing de recibos | CORD | Tree-Edit-Distance Acc. | ~91.6 |
| Parsing de recibos | SROIE | F1 | ~83.2 |
| Comprensión de formularios | FUNSD | F1 | según setup |
| Clasificación | RVL-CDIP | Accuracy | ~95.3 |

Donut es competitivo con métodos basados en OCR, pero **más rápido y sin depender
de un OCR externo**.""")

# ------------------------------------------------------------------ Sección 9
md(r"""## 9. Conclusión y preguntas frecuentes

**Lo aprendido:** Donut mapea imagen → texto/JSON de punta a punta, con un
encoder Swin que "ve" la imagen, un decoder Transformer que genera la salida y
**cross-attention** como puente entre ambos.

**FAQ**
1. **¿Por qué no necesita OCR externo?** El encoder de visión lee la imagen
   directamente; el reconocimiento de contenido es parte del propio modelo.
2. **¿Cómo "ve" la imagen un transformer de texto?** No la ve el decoder de texto
   por sí solo: el encoder de visión la convierte en vectores y el decoder los
   consulta vía cross-attention.
3. **¿Qué es el cross-attention y por qué importa?** Es la capa del decoder que
   atiende a los vectores del encoder; es la conexión imagen↔texto.
4. **¿Cómo se entrena sin texto OCR?** Se entrena con pares (imagen, secuencia
   objetivo). La pérdida es cross-entropy sobre los tokens objetivo; nunca se
   pasa texto OCR.
5. **¿Diferencia encoder vs decoder?** El encoder codifica la imagen en vectores;
   el decoder genera la secuencia de salida token por token usando esos vectores.
6. **¿Qué salida genera?** Una secuencia de tokens que se decodifica a texto
   plano o a **JSON estructurado** (con `token2json`).

**Referencias**
- Paper: https://arxiv.org/abs/2111.15664
- Código: https://github.com/clovaai/donut
- Modelo: https://huggingface.co/naver-clova-ix/donut-base
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open("Donut_Colab_Notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook escrito. Celdas:", len(cells))
