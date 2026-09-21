# Donut — OCR-free Document Understanding Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/restrepogerard/donut-document-understanding/blob/main/Donut_Colab_Notebook.ipynb)

Proyecto educativo universitario sobre el paper **"Donut: OCR-free Document Understanding Transformer"** (ECCV 2022, [arXiv:2111.15664](https://arxiv.org/abs/2111.15664)).

El objetivo es explicar de forma clara y didáctica cómo opera Donut: su arquitectura completa (Swin Transformer + Transformer decoder), el flujo de datos y el rol del **cross-attention** que conecta encoder y decoder.

---

## 🎯 ¿Qué hace Donut?

Donut convierte una **imagen de documento** directamente en **texto/JSON estructurado**, sin necesidad de un motor OCR externo. El modelo "lee" la imagen mediante un encoder de visión y genera la respuesta token por token con un decoder de texto.

```
Pipeline tradicional:   Imagen → OCR externo → Texto → Modelo NLP → Salida
Pipeline Donut:         Imagen → Encoder (Swin) → Cross-Attention → Decoder → Salida
```

---

## 🧠 Arquitectura en una frase

| Componente        | Rol                                                                 |
|-------------------|---------------------------------------------------------------------|
| **Encoder**       | Swin Transformer: imagen → secuencia de vectores visuales `(N, D)`  |
| **Decoder**       | Transformer decoder autoregresivo: genera tokens uno a uno          |
| **Cross-Attention** | Conecta ambos: el decoder "mira" los vectores del encoder al generar cada token |
| **Sin OCR**       | La imagen entra directo al modelo, sin paso de reconocimiento de texto |

---

## 📁 Estructura del proyecto

| Archivo | Descripción |
|---------|-------------|
| `Donut_Colab_Notebook.ipynb` | **Notebook principal** para Google Colab. Explica la arquitectura y ejecuta inferencia completa (secciones 1–9). |
| `Ejemplo_de_implementacion_Donut.ipynb` | Notebook mínimo de inferencia sobre una imagen (`image.png`). |
| `Donut_Presentacion.pptx` | Presentación (17 slides) que explica el modelo de entrada a salida, con foco en el cross-attention. |
| `01_inferencia_matriculas.py` | Demo de inferencia sobre una imagen sintética (estilo matrícula/documento). |
| `02_arquitectura_explicada.py` | Explora encoder, decoder, cross-attention y las formas de los tensores en cada etapa. |
| `03_ejemplo_entrenamiento.py` | Crea facturas sintéticas, arma un `Dataset` y muestra el bucle de fine-tuning. |
| `04_visualizar_cross_attention.py` | Genera un mapa de calor del cross-attention superpuesto sobre la imagen. |
| `image.png` | Imagen de ejemplo (factura/recibo real) para probar la inferencia. |
| `_build_notebook.py` / `_build_pptx.py` | Generadores del notebook y de la presentación. |
| `requirements.txt` | Dependencias del proyecto. |

---

## 🚀 Cómo usar (para cualquier usuario)

### Opción 1 — Google Colab (recomendado, con GPU gratis)
1. Haz clic en el badge **Open In Colab** de arriba (abre `Donut_Colab_Notebook.ipynb` directo desde este repo).
2. Activa GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU`.
3. Ejecuta la celda de instalación. **Reinicia el entorno** cuando lo pida y continúa desde la celda de imports.
4. Para usar tu propia imagen: súbela al panel `📁 Archivos` de Colab y pon su ruta en `IMAGE_PATH` (ej. `/content/mi_factura.png`). Por defecto usa `image.png` del repo.
5. Ejecuta el resto de celdas en orden.

### Opción 2 — Local
```bash
git clone https://github.com/restrepogerard/donut-document-understanding.git
cd TU_REPO
pip install -r requirements.txt
# el tokenizer de Donut necesita protobuf + sentencepiece (ya en requirements)
python 01_inferencia_matriculas.py
python 02_arquitectura_explicada.py
python 03_ejemplo_entrenamiento.py
python 04_visualizar_cross_attention.py
```

> **Nota:** la primera ejecución descarga el modelo (~800 MB) desde Hugging Face.
> En CPU la inferencia funciona pero es lenta; con GPU es casi inmediata.

---

## 📊 Resultados del paper

| Tarea   | Dataset | Métrica                | Donut  |
|---------|---------|------------------------|--------|
| DocVQA  | DocVQA  | ANLS                   | ~67.5  |
| Recibos | CORD    | Tree-Edit-Distance Acc | ~91.6  |
| Recibos | SROIE   | F1                     | ~83.2  |
| Formularios | FUNSD | F1                   | ~11–14 (setup específico) |

> Los números exactos dependen del setup y del checkpoint. Ver el paper para detalles.

---

## 📚 Referencias

- Paper: <https://arxiv.org/abs/2111.15664>
- Código oficial: <https://github.com/clovaai/donut>
- Modelo Hugging Face: <https://huggingface.co/naver-clova-ix/donut-base>

**Modelo usado:** `naver-clova-ix/donut-base-finetuned-cord-v2` — Donut fine-tuneado sobre CORD (Swin Transformer encoder + BART-like transformer decoder). Al estar fine-tuneado para parsing de recibos, produce **JSON estructurado real**.

> El notebook instala `transformers` desde el repo oficial de Hugging Face más `accelerate` y `timm` (el encoder Swin depende de `timm`), tal como en el ejemplo probado en Colab.
