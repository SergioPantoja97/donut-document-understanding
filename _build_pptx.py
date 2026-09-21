"""
Genera 'Donut_Presentacion.pptx' — presentación educativa del modelo Donut
(OCR-free Document Understanding Transformer), enfocada en explicar el modelo
de entrada a salida, con énfasis en el cross-attention.

Uso: python3 _build_pptx.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------------------------------------------------------------- paleta
AZUL_OSCURO = RGBColor(0x0B, 0x2E, 0x59)   # títulos / cabeceras
AZUL = RGBColor(0x1F, 0x6F, 0xB2)          # acentos
CELESTE = RGBColor(0xE8, 0xF1, 0xFA)       # fondos suaves
NARANJA = RGBColor(0xE8, 0x7A, 0x1E)       # resaltar (cross-attention)
GRIS = RGBColor(0x3C, 0x3C, 0x3C)          # texto
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
VERDE = RGBColor(0x2E, 0x8B, 0x57)

prs = Presentation()
prs.slide_width = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def add_slide():
    return prs.slides.add_slide(BLANK)


def bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def box(slide, x, y, w, h):
    return slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))


def set_text(tf, text, size=18, color=GRIS, bold=False, align=PP_ALIGN.LEFT,
             font="Calibri", bullet=False):
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = ("•  " + text) if bullet else text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def add_para(tf, text, size=18, color=GRIS, bold=False, align=PP_ALIGN.LEFT,
             bullet=False, level=0, space_before=6, font="Calibri"):
    p = tf.add_paragraph()
    p.alignment = align
    p.level = level
    p.space_before = Pt(space_before)
    prefix = "•  " if bullet else ""
    r = p.add_run()
    r.text = prefix + text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def header(slide, titulo, num=None):
    """Barra superior con título."""
    bar = slide.shapes.add_shape(1, 0, 0, SW, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = AZUL_OSCURO
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.4)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(tf, titulo, size=28, color=BLANCO, bold=True)
    if num is not None:
        nb = box(slide, 12.4, 0.28, 0.8, 0.5)
        set_text(nb.text_frame, str(num), size=18, color=CELESTE, bold=True,
                 align=PP_ALIGN.RIGHT)


def rect(slide, x, y, w, h, fill, line=None, text=None, size=14,
         tcolor=GRIS, bold=False, talign=PP_ALIGN.CENTER):
    sp = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1.5)
    if text is not None:
        tf = sp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_text(tf, text, size=size, color=tcolor, bold=bold, align=talign)
    return sp


def arrow_down(slide, x, y, w=0.5, h=0.4, color=AZUL):
    sp = slide.shapes.add_shape(67, Inches(x), Inches(y), Inches(w), Inches(h))  # DOWN_ARROW
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    return sp


def arrow_right(slide, x, y, w=0.6, h=0.45, color=AZUL):
    sp = slide.shapes.add_shape(66, Inches(x), Inches(y), Inches(w), Inches(h))  # RIGHT_ARROW
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    return sp


# ============================================================== SLIDE 1 — Portada
s = add_slide()
bg(s, AZUL_OSCURO)
t = box(s, 0.8, 2.1, 11.7, 2.0)
set_text(t.text_frame, "🍩 Donut", size=60, color=BLANCO, bold=True)
add_para(t.text_frame, "OCR-free Document Understanding Transformer",
         size=30, color=CELESTE, bold=True, space_before=8)
sub = box(s, 0.8, 4.3, 11.7, 2.2)
set_text(sub.text_frame, "Cómo funciona el modelo: de la imagen de entrada al JSON de salida",
         size=22, color=BLANCO)
add_para(sub.text_frame, "Paper: Kim et al., ECCV 2022  ·  arXiv:2111.15664",
         size=16, color=CELESTE, space_before=16)
add_para(sub.text_frame, "naver-clova-ix/donut-base  ·  Swin Transformer + Transformer Decoder",
         size=16, color=CELESTE, space_before=4)

# ============================================================== SLIDE 2 — Agenda
s = add_slide(); bg(s, BLANCO)
header(s, "Contenido", 2)
t = box(s, 0.8, 1.4, 11.7, 5.6)
set_text(t.text_frame, "1.  El problema: OCR tradicional vs Donut", size=22, color=AZUL_OSCURO, bold=True)
for txt in [
    "2.  Entrada (Input): la imagen del documento",
    "3.  Encoder — Swin Transformer",
    "4.  Decoder — Transformer autoregresivo",
    "5.  Cross-Attention (el punto clave)",
    "6.  Flujo completo: imagen → encoder → cross-attn → decoder → JSON",
    "7.  Salida (Output): texto / JSON estructurado",
    "8.  Entrenamiento y pre-entrenamiento",
    "9.  Resultados y tipos de documentos",
    "10. Ventajas, limitaciones y conclusión",
]:
    add_para(t.text_frame, txt, size=22, color=GRIS, space_before=10)

# ============================================================== SLIDE 3 — Problema
s = add_slide(); bg(s, BLANCO)
header(s, "1. El problema: OCR tradicional vs Donut", 3)
# Pipeline tradicional
rect(s, 0.6, 1.5, 12.1, 0.5, CELESTE, text="PIPELINE TRADICIONAL (con OCR externo)",
     size=16, tcolor=AZUL_OSCURO, bold=True)
rect(s, 0.8, 2.15, 2.3, 0.9, BLANCO, line=AZUL, text="Imagen", size=15)
arrow_right(s, 3.2, 2.4)
rect(s, 3.9, 2.15, 2.3, 0.9, RGBColor(0xF6,0xD6,0xD6), line=RGBColor(0xC0,0x39,0x39),
     text="OCR externo\n(reconoce texto)", size=13)
arrow_right(s, 6.3, 2.4)
rect(s, 7.0, 2.15, 2.0, 0.9, BLANCO, line=AZUL, text="Texto", size=15)
arrow_right(s, 9.1, 2.4)
rect(s, 9.7, 2.15, 3.0, 0.9, BLANCO, line=AZUL, text="Modelo NLP → Salida", size=13)
n = box(s, 0.8, 3.1, 12, 0.5)
set_text(n.text_frame, "⚠️  Los errores del OCR se propagan · más lento · más caro · pipeline en 2 etapas",
         size=14, color=RGBColor(0xC0,0x39,0x39))

# Pipeline Donut
rect(s, 0.6, 4.1, 12.1, 0.5, CELESTE, text="PIPELINE DONUT (OCR-free, end-to-end)",
     size=16, tcolor=VERDE, bold=True)
rect(s, 0.8, 4.75, 2.3, 0.9, BLANCO, line=AZUL, text="Imagen", size=15)
arrow_right(s, 3.2, 5.0)
rect(s, 3.9, 4.75, 2.6, 0.9, CELESTE, line=AZUL, text="Encoder\n(Swin)", size=14, bold=True)
arrow_right(s, 6.6, 5.0, color=NARANJA)
rect(s, 7.3, 4.75, 2.6, 0.9, RGBColor(0xFC,0xE4,0xCB), line=NARANJA,
     text="Cross-Attention", size=13, bold=True, tcolor=NARANJA)
arrow_right(s, 10.0, 5.0)
rect(s, 10.6, 4.75, 2.1, 0.9, CELESTE, line=AZUL, text="Decoder\n→ JSON", size=14, bold=True)
n2 = box(s, 0.8, 5.75, 12, 0.6)
set_text(n2.text_frame, "✅  La imagen entra directo al modelo · sin OCR · entrenado de punta a punta",
         size=14, color=VERDE)

# ============================================================== SLIDE 4 — Entrada
s = add_slide(); bg(s, BLANCO)
header(s, "2. Entrada (Input)", 4)
rect(s, 4.6, 1.5, 4.1, 2.3, CELESTE, line=AZUL,
     text="🧾\nImagen del documento\n(factura, matrícula,\nrecibo, formulario)",
     size=18, tcolor=AZUL_OSCURO, bold=True)
t = box(s, 0.8, 4.1, 11.7, 3.0)
set_text(t.text_frame, "La imagen del documento (por ejemplo 2560×1920 px, RGB)",
         size=20, color=GRIS, bullet=True)
add_para(t.text_frame, "NO requiere preprocesamiento externo ni un formato específico",
         size=20, color=GRIS, bullet=True)
add_para(t.text_frame, "NO se ejecuta OCR previo: la imagen va directamente al modelo",
         size=20, color=GRIS, bullet=True)
add_para(t.text_frame, "El processor de Donut solo redimensiona y normaliza la imagen → pixel_values",
         size=20, color=GRIS, bullet=True)
add_para(t.text_frame, "Ejemplo de tensor de entrada: (batch, 3 canales, alto, ancho)",
         size=18, color=AZUL, bullet=True, space_before=10)

# ============================================================== SLIDE 5 — Encoder
s = add_slide(); bg(s, BLANCO)
header(s, "3. Encoder — Swin Transformer", 5)
t = box(s, 0.7, 1.3, 6.2, 5.8)
set_text(t.text_frame, "¿Qué es?", size=22, color=AZUL_OSCURO, bold=True)
add_para(t.text_frame, "Un Swin Transformer que actúa como vision encoder: convierte la imagen en una secuencia de vectores visuales.",
         size=18, color=GRIS, space_before=6)
add_para(t.text_frame, "¿Cómo funciona?", size=22, color=AZUL_OSCURO, bold=True, space_before=14)
add_para(t.text_frame, "Divide la imagen en patches (patch embedding)", size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "4 stages jerárquicos con capas de atención", size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Shifted windows: atención por ventanas desplazadas (eficiente)", size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Salida: secuencia de vectores de dimensión D", size=18, color=AZUL, bullet=True)

# Diagrama de reducción jerárquica a la derecha
rect(s, 7.4, 1.5, 5.2, 0.5, AZUL_OSCURO, text="Reducción jerárquica (4 stages)", size=14, tcolor=BLANCO, bold=True)
stages = [
    ("Imagen → patches → embeddings", CELESTE),
    ("Stage 1", CELESTE),
    ("Stage 2 (menos vectores, más dim)", CELESTE),
    ("Stage 3", CELESTE),
    ("Stage 4 → secuencia (N, D)", RGBColor(0xCF,0xE6,0xD4)),
]
y = 2.2
for txt, col in stages:
    rect(s, 7.7, y, 4.6, 0.62, col, line=AZUL, text=txt, size=13)
    if y < 5.0:
        arrow_down(s, 9.8, y + 0.63, w=0.35, h=0.28)
    y += 0.95

# ============================================================== SLIDE 6 — Decoder
s = add_slide(); bg(s, BLANCO)
header(s, "4. Decoder — Transformer autoregresivo", 6)
t = box(s, 0.7, 1.3, 6.2, 5.8)
set_text(t.text_frame, "¿Qué es?", size=22, color=AZUL_OSCURO, bold=True)
add_para(t.text_frame, "Un Transformer decoder estándar que genera la salida token por token.",
         size=18, color=GRIS, space_before=6)
add_para(t.text_frame, "¿Cómo funciona?", size=22, color=AZUL_OSCURO, bold=True, space_before=14)
add_para(t.text_frame, "Genera texto de forma autoregresiva (un token a la vez)", size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Positional encoding para conocer la posición de cada token", size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Self-attention sobre los tokens ya generados", size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Cross-attention sobre la salida del encoder (la imagen)", size=18, color=NARANJA, bullet=True, bold=True)
add_para(t.text_frame, "Está condicionado al encoder: genera según la representación visual",
         size=18, color=AZUL, bullet=True, space_before=8)

# Bloque de una capa del decoder
rect(s, 7.5, 1.5, 5.0, 0.55, AZUL_OSCURO, text="Una capa del decoder", size=15, tcolor=BLANCO, bold=True)
rect(s, 7.7, 2.25, 4.6, 0.7, CELESTE, line=AZUL, text="1) Self-Attention (tokens previos)", size=13)
arrow_down(s, 9.8, 2.98, w=0.35, h=0.25)
rect(s, 7.7, 3.35, 4.6, 0.7, RGBColor(0xFC,0xE4,0xCB), line=NARANJA,
     text="2) CROSS-ATTENTION (mira la imagen)", size=13, bold=True, tcolor=NARANJA)
arrow_down(s, 9.8, 4.08, w=0.35, h=0.25)
rect(s, 7.7, 4.45, 4.6, 0.7, CELESTE, line=AZUL, text="3) Feed-Forward (FFN)", size=13)
arrow_down(s, 9.8, 5.18, w=0.35, h=0.25)
rect(s, 7.7, 5.55, 4.6, 0.7, RGBColor(0xCF,0xE6,0xD4), line=AZUL, text="→ Predice el siguiente token", size=13, bold=True)

# ============================================================== SLIDE 7 — Cross-attention (concepto)
s = add_slide(); bg(s, BLANCO)
header(s, "5. Cross-Attention — el punto clave", 7)
# Decoder box
rect(s, 0.8, 1.6, 3.6, 1.0, CELESTE, line=AZUL, text="DECODER (texto)\nToken en curso: \"total\"", size=14, bold=True, tcolor=AZUL_OSCURO)
# Encoder box
rect(s, 8.9, 1.6, 3.6, 1.0, CELESTE, line=AZUL, text="ENCODER (imagen)\nVectores visuales", size=14, bold=True, tcolor=AZUL_OSCURO)
# Q / K,V
rect(s, 0.8, 3.0, 3.6, 0.9, RGBColor(0xFC,0xE4,0xCB), line=NARANJA, text="Query (Q)\n(del decoder)", size=14, bold=True, tcolor=NARANJA)
rect(s, 8.9, 3.0, 3.6, 0.9, RGBColor(0xFC,0xE4,0xCB), line=NARANJA, text="Key (K) + Value (V)\n(del encoder)", size=14, bold=True, tcolor=NARANJA)
# Center attention
rect(s, 4.9, 3.0, 3.4, 0.9, NARANJA, text="ATENCIÓN\nQ · K → pesos → V", size=15, bold=True, tcolor=BLANCO)
arrow_right(s, 4.45, 3.15, w=0.5, color=NARANJA)
arrow_right(s, 8.35, 3.15, w=0.5, color=NARANJA)  # visual pointing (K,V into attn from right)
# Context result
rect(s, 4.9, 4.3, 3.4, 0.9, RGBColor(0xCF,0xE6,0xD4), line=AZUL, text="Context vector\n(qué mirar en la imagen)", size=14, bold=True, tcolor=AZUL_OSCURO)
arrow_down(s, 6.4, 3.92, w=0.4, h=0.35, color=AZUL)
# Explanation
t = box(s, 0.8, 5.5, 11.7, 1.7)
set_text(t.text_frame, "El decoder \"pregunta\" a la imagen qué mirar (Query) y el encoder \"responde\" con la información visual relevante (Key, Value).",
         size=17, color=GRIS)
add_para(t.text_frame, "→ Es la conexión que permite al decoder ENFOCARSE en la parte de la imagen relevante para el token que va a generar.",
         size=17, color=NARANJA, bold=True, space_before=6)

# ============================================================== SLIDE 8 — Cross-attention (ejemplo)
s = add_slide(); bg(s, CELESTE)
header(s, "5. Cross-Attention — ejemplo intuitivo", 8)
rect(s, 1.2, 1.6, 10.9, 1.5, BLANCO, line=NARANJA,
     text="El decoder quiere generar el token  \"total: $50.00\"", size=24, bold=True, tcolor=AZUL_OSCURO)
t = box(s, 1.2, 3.4, 10.9, 3.6)
set_text(t.text_frame, "1)  El decoder lanza una \"pregunta\" (Query): ¿dónde está el total?",
         size=20, color=GRIS, bullet=True)
add_para(t.text_frame, "2)  La imagen responde (Key, Value): estos vectores corresponden a la zona del total",
         size=20, color=GRIS, bullet=True, space_before=10)
add_para(t.text_frame, "3)  La atención cruzada hace que el modelo se ENFOQUE en esa región",
         size=20, color=GRIS, bullet=True, space_before=10)
add_para(t.text_frame, "4)  Con ese contexto visual, el decoder genera el token correcto: \"50.00\"",
         size=20, color=VERDE, bullet=True, bold=True, space_before=10)
add_para(t.text_frame, "Sin cross-attention, el decoder generaría texto \"a ciegas\", sin mirar la imagen.",
         size=17, color=NARANJA, bold=True, space_before=14)

# ============================================================== SLIDE 9 — Flujo completo
s = add_slide(); bg(s, BLANCO)
header(s, "6. Flujo completo del modelo", 9)
steps = [
    ("1. IMAGEN DE ENTRADA  (ej. 2560×1920 px, RGB)", CELESTE, AZUL),
    ("2. PATCHES  →  la imagen se divide en parches (ej. 16×16 px)", CELESTE, AZUL),
    ("3. EMBEDDINGS  →  cada patch se proyecta a un vector (ej. dim 768)", CELESTE, AZUL),
    ("4. ENCODER (Swin, 4 stages)  →  secuencia de vectores (N, D)", RGBColor(0xD6,0xE7,0xF7), AZUL),
    ("5. CROSS-ATTENTION  →  el decoder consulta los vectores de la imagen", RGBColor(0xFC,0xE4,0xCB), NARANJA),
    ("6. DECODER (self-attn + cross-attn + FFN)  →  genera un token", RGBColor(0xD6,0xE7,0xF7), AZUL),
    ("7. REPITE autoregresivamente hasta <eos>  →  JSON estructurado", RGBColor(0xCF,0xE6,0xD4), VERDE),
]
y = 1.35
for txt, fill, line in steps:
    bold = ("CROSS-ATTENTION" in txt)
    tcolor = NARANJA if bold else GRIS
    rect(s, 1.3, y, 10.7, 0.62, fill, line=line, text=txt, size=15,
         bold=bold, tcolor=tcolor, talign=PP_ALIGN.LEFT)
    if y < 6.0:
        arrow_down(s, 6.4, y + 0.63, w=0.35, h=0.18, color=line)
    y += 0.82

# ============================================================== SLIDE 10 — Salida
s = add_slide(); bg(s, BLANCO)
header(s, "7. Salida (Output)", 10)
t = box(s, 0.8, 1.35, 11.7, 1.6)
set_text(t.text_frame, "El modelo produce una secuencia de tokens que se decodifica a texto o a JSON estructurado.",
         size=20, color=GRIS)
add_para(t.text_frame, "Puede usarse como texto libre o como JSON con campos (clave: valor).",
         size=18, color=AZUL, space_before=6)
# Ejemplo JSON
rect(s, 1.5, 3.1, 10.3, 3.2, RGBColor(0x1E,0x1E,0x1E),
     text='{\n   "placa": "A123BC",\n   "provincia": "DN",\n   "anio": "2020"\n}',
     size=22, tcolor=RGBColor(0x9C,0xDC,0xFE), talign=PP_ALIGN.LEFT)
n = box(s, 1.5, 6.4, 10.3, 0.7)
set_text(n.text_frame, "Ejemplo: de una imagen de matrícula → JSON con placa, provincia y año.",
         size=16, color=GRIS)

# ============================================================== SLIDE 11 — Entrenamiento
s = add_slide(); bg(s, BLANCO)
header(s, "8. Entrenamiento y pre-entrenamiento", 11)
t = box(s, 0.8, 1.4, 11.7, 5.6)
set_text(t.text_frame, "Pre-entrenamiento", size=22, color=AZUL_OSCURO, bold=True)
add_para(t.text_frame, "Tarea \"leer\" documentos: aprende a transcribir texto a partir de imágenes (documentos sintéticos + reales con sus transcripciones).",
         size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Fine-tuning", size=22, color=AZUL_OSCURO, bold=True, space_before=14)
add_para(t.text_frame, "Se adapta a una tarea concreta (facturas, matrículas, recibos) con pares imagen → salida estructurada.",
         size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Objetivo de entrenamiento", size=22, color=AZUL_OSCURO, bold=True, space_before=14)
add_para(t.text_frame, "Cross-entropy sobre los tokens generados (teacher forcing).",
         size=18, color=GRIS, bullet=True)
add_para(t.text_frame, "Clave: NUNCA se le pasa texto OCR. Solo se le da la IMAGEN y la SECUENCIA OBJETIVO; el modelo aprende el mapeo imagen → texto/JSON de punta a punta.",
         size=18, color=NARANJA, bold=True, bullet=True, space_before=10)

# ============================================================== SLIDE 12 — Resultados / tipos
s = add_slide(); bg(s, BLANCO)
header(s, "9. Resultados y tipos de documentos", 12)
# Tabla de resultados (izquierda)
lt = box(s, 0.7, 1.25, 5.9, 0.5)
set_text(lt.text_frame, "Resultados del paper (aprox.)", size=18, color=AZUL_OSCURO, bold=True)
rows = [("Tarea / Dataset", "Métrica", "Donut"),
        ("DocVQA", "ANLS", "~67.5"),
        ("CORD (recibos)", "TED-Acc.", "~91.6"),
        ("SROIE (recibos)", "F1", "~83.2"),
        ("RVL-CDIP (clasif.)", "Accuracy", "~95.3")]
tbl = s.shapes.add_table(len(rows), 3, Inches(0.7), Inches(1.75),
                         Inches(5.9), Inches(2.7)).table
for j, val in enumerate(rows[0]):
    c = tbl.cell(0, j); c.text = val
    c.fill.solid(); c.fill.fore_color.rgb = AZUL_OSCURO
    for para in c.text_frame.paragraphs:
        para.runs[0].font.color.rgb = BLANCO
        para.runs[0].font.bold = True
        para.runs[0].font.size = Pt(13)
for i in range(1, len(rows)):
    for j, val in enumerate(rows[i]):
        c = tbl.cell(i, j); c.text = val
        c.fill.solid(); c.fill.fore_color.rgb = CELESTE if i % 2 else BLANCO
        for para in c.text_frame.paragraphs:
            para.runs[0].font.size = Pt(13)
            para.runs[0].font.color.rgb = GRIS

# Tabla de tipos de documentos (derecha)
rt = box(s, 6.9, 1.25, 5.9, 0.5)
set_text(rt.text_frame, "Tipos de documentos", size=18, color=AZUL_OSCURO, bold=True)
rows2 = [("Documento", "Qué extrae"),
         ("Matrícula", "Placa, provincia, año"),
         ("Factura", "Total, fecha, cliente"),
         ("Ticket", "Productos, total, fecha"),
         ("Formulario", "Campos estructurados")]
tbl2 = s.shapes.add_table(len(rows2), 2, Inches(6.9), Inches(1.75),
                          Inches(5.9), Inches(2.7)).table
for j, val in enumerate(rows2[0]):
    c = tbl2.cell(0, j); c.text = val
    c.fill.solid(); c.fill.fore_color.rgb = AZUL_OSCURO
    for para in c.text_frame.paragraphs:
        para.runs[0].font.color.rgb = BLANCO
        para.runs[0].font.bold = True
        para.runs[0].font.size = Pt(13)
for i in range(1, len(rows2)):
    for j, val in enumerate(rows2[i]):
        c = tbl2.cell(i, j); c.text = val
        c.fill.solid(); c.fill.fore_color.rgb = CELESTE if i % 2 else BLANCO
        for para in c.text_frame.paragraphs:
            para.runs[0].font.size = Pt(13)
            para.runs[0].font.color.rgb = GRIS

n = box(s, 0.7, 4.8, 12, 1.8)
set_text(n.text_frame, "Donut es competitivo con métodos basados en OCR, pero más rápido y sin depender de un motor OCR externo.",
         size=17, color=GRIS)
add_para(n.text_frame, "Los valores exactos dependen del setup y del checkpoint usado.",
         size=14, color=AZUL, space_before=6)

# ============================================================== SLIDE 13 — Ventajas / limitaciones
s = add_slide(); bg(s, BLANCO)
header(s, "10. Ventajas, limitaciones y conclusión", 13)
lt = box(s, 0.7, 1.35, 5.9, 5.5)
set_text(lt.text_frame, "Ventajas", size=22, color=VERDE, bold=True)
for txt in ["Sin OCR externo: menos errores encadenados",
            "End-to-end, más simple de mantener",
            "Salida directa en JSON estructurado",
            "Multilingüe (no depende de OCR por idioma)",
            "Más rápido en inferencia que pipelines con OCR"]:
    add_para(lt.text_frame, txt, size=18, color=GRIS, bullet=True, space_before=8)

rt = box(s, 6.9, 1.35, 5.9, 5.5)
set_text(rt.text_frame, "Limitaciones", size=22, color=RGBColor(0xC0,0x39,0x39), bold=True)
for txt in ["Sensible a imágenes de muy baja calidad",
            "Requiere fine-tuning por tipo de documento",
            "Costo de entrenamiento alto",
            "Puede fallar con layouts muy fuera de distribución"]:
    add_para(rt.text_frame, txt, size=18, color=GRIS, bullet=True, space_before=8)

# ============================================================== SLIDE (valor) — Visualización cross-attention
s = add_slide(); bg(s, BLANCO)
header(s, "Valor agregado: visualizar el cross-attention", 14)
t = box(s, 0.8, 1.3, 11.7, 1.5)
set_text(t.text_frame, "Superponemos un mapa de calor sobre la imagen para ver QUÉ mira el modelo al generar la salida.",
         size=19, color=GRIS)
add_para(t.text_frame, "Se promedian los pesos de cross-attention (pasos × capas × cabezas) y se reproyectan a la cuadrícula de patches.",
         size=16, color=AZUL, space_before=6)
# Tres paneles ilustrativos
rect(s, 0.9, 3.0, 3.6, 3.0, BLANCO, line=AZUL, text="1) Imagen\nde entrada", size=16, tcolor=AZUL_OSCURO, bold=True)
rect(s, 4.85, 3.0, 3.6, 3.0, RGBColor(0xFC,0xE4,0xCB), line=NARANJA,
     text="2) Mapa de\ncross-attention\n(zonas cálidas =\nmás atención)", size=15, tcolor=NARANJA, bold=True)
rect(s, 8.8, 3.0, 3.6, 3.0, RGBColor(0xCF,0xE6,0xD4), line=VERDE,
     text="3) Overlay\n(atención sobre\nla imagen)", size=16, tcolor=VERDE, bold=True)
n = box(s, 0.8, 6.2, 11.7, 1.0)
set_text(n.text_frame, "→ Evidencia visual de que el cross-attention conecta cada token generado con la región relevante de la imagen.",
         size=16, color=GRIS)
add_para(n.text_frame, "Generado por: 04_visualizar_cross_attention.py  (y sección 7.3 del notebook).",
         size=13, color=AZUL, space_before=4)

# ============================================================== SLIDE (valor) — Demo multi-documento
s = add_slide(); bg(s, BLANCO)
header(s, "Valor agregado: experimentos con distintos documentos", 15)
rows = [("Documento", "Ejemplo", "Qué extrae Donut"),
        ("Matrícula", "Placa A123BC", "Placa, provincia, año"),
        ("Factura", "Factura #123", "Total, fecha, cliente"),
        ("Ticket", "Ticket #456", "Productos, total, fecha"),
        ("Formulario", "Formulario X", "Campos estructurados"),
        ("Pasaporte", "Documento ID", "Información personal")]
tbl = s.shapes.add_table(len(rows), 3, Inches(1.0), Inches(1.5),
                         Inches(11.3), Inches(3.9)).table
for j, val in enumerate(rows[0]):
    c = tbl.cell(0, j); c.text = val
    c.fill.solid(); c.fill.fore_color.rgb = AZUL_OSCURO
    for para in c.text_frame.paragraphs:
        para.runs[0].font.color.rgb = BLANCO
        para.runs[0].font.bold = True
        para.runs[0].font.size = Pt(16)
for i in range(1, len(rows)):
    for j, val in enumerate(rows[i]):
        c = tbl.cell(i, j); c.text = val
        c.fill.solid(); c.fill.fore_color.rgb = CELESTE if i % 2 else BLANCO
        for para in c.text_frame.paragraphs:
            para.runs[0].font.size = Pt(15)
            para.runs[0].font.color.rgb = GRIS
n = box(s, 1.0, 5.6, 11.3, 1.4)
set_text(n.text_frame, "El mismo modelo, distintas salidas: Donut adapta la extracción al tipo de documento (con fine-tuning por dominio).",
         size=17, color=GRIS)
add_para(n.text_frame, "Idea de experimento: pasar una factura, una matrícula y un ticket, y comparar el JSON extraído en cada caso.",
         size=15, color=AZUL, space_before=6)

# ============================================================== SLIDE (valor) — Análisis de errores
s = add_slide(); bg(s, BLANCO)
header(s, "Valor agregado: análisis de errores y comparación", 16)
lt = box(s, 0.7, 1.35, 5.9, 5.6)
set_text(lt.text_frame, "¿Cuándo falla Donut?", size=22, color=RGBColor(0xC0,0x39,0x39), bold=True)
for txt in ["Imágenes de muy baja calidad o borrosas",
            "Layouts muy distintos a los del entrenamiento",
            "Idiomas/formatos no vistos en fine-tuning",
            "Campos ambiguos o texto manuscrito difícil"]:
    add_para(lt.text_frame, txt, size=18, color=GRIS, bullet=True, space_before=8)

rt = box(s, 6.9, 1.35, 5.9, 5.6)
set_text(rt.text_frame, "Donut vs OCR tradicional", size=22, color=AZUL_OSCURO, bold=True)
for txt in ["OCR: 2 etapas, errores encadenados",
            "Donut: end-to-end, sin propagación de errores de OCR",
            "OCR: más rápido de adaptar sin datos",
            "Donut: mejor en documentos estructurados tras fine-tuning",
            "Donut: multilingüe sin cambiar el motor de OCR"]:
    add_para(rt.text_frame, txt, size=17, color=GRIS, bullet=True, space_before=8)

# ============================================================== SLIDE 14 — Cierre
s = add_slide(); bg(s, AZUL_OSCURO)
t = box(s, 0.9, 1.6, 11.5, 4.8)
set_text(t.text_frame, "En una frase", size=26, color=NARANJA, bold=True)
add_para(t.text_frame, "Donut lee una imagen de documento y genera JSON estructurado, sin OCR: el encoder Swin \"ve\" la imagen, el decoder genera el texto, y el cross-attention es el puente que deja al decoder mirar la imagen al escribir cada token.",
         size=24, color=BLANCO, space_before=18)
add_para(t.text_frame, "Imagen → Encoder → Cross-Attention → Decoder → JSON",
         size=22, color=CELESTE, bold=True, space_before=24, align=PP_ALIGN.CENTER)
ref = box(s, 0.9, 6.4, 11.5, 0.9)
set_text(ref.text_frame, "Referencias:  arXiv:2111.15664  ·  github.com/clovaai/donut  ·  huggingface.co/naver-clova-ix/donut-base",
         size=14, color=CELESTE)

# ---------------------------------------------------------------- guardar
prs.save("Donut_Presentacion.pptx")
print("Presentación creada: Donut_Presentacion.pptx  ·  slides:", len(prs.slides._sldIdLst))
