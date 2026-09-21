"""
02_arquitectura_explicada.py
=============================

Explora la ARQUITECTURA de Donut paso a paso, inspeccionando las formas
(shapes) de los tensores en cada etapa del flujo:

    Imagen -> Encoder (Swin) -> vectores -> Cross-Attention -> Decoder -> tokens

Objetivo educativo: entender QUÉ hace cada componente y CÓMO cambian las
dimensiones de los datos a lo largo del modelo.

Ejecución:
    python 02_arquitectura_explicada.py
"""

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_NAME = "naver-clova-ix/donut-base"


def separador(titulo: str):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # -----------------------------------------------------------------
    # Cargar modelo
    # -----------------------------------------------------------------
    separador("1. CARGA DEL MODELO")
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    enc_cfg = model.config.encoder
    dec_cfg = model.config.decoder
    print(f"Encoder : {enc_cfg.model_type}  (Swin Transformer)")
    print(f"Decoder : {dec_cfg.model_type}  (Transformer decoder autoregresivo)")
    print(f"Tamaño de imagen esperado : {enc_cfg.image_size}")
    print(f"Dimensión oculta encoder  : {getattr(enc_cfg, 'hidden_size', 'n/a')}")
    print(f"Dimensión oculta decoder  : {dec_cfg.hidden_size}")
    print(f"Vocabulario decoder       : {dec_cfg.vocab_size}")
    print(f"Parámetros totales        : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # -----------------------------------------------------------------
    # Preparar entrada
    # -----------------------------------------------------------------
    separador("2. IMAGEN DE ENTRADA -> pixel_values")
    imagen = Image.new("RGB", (1280, 960), color=(255, 255, 255))
    pixel_values = processor(imagen, return_tensors="pt").pixel_values.to(device)
    print(f"pixel_values.shape = {tuple(pixel_values.shape)}")
    print("  -> (batch, canales, alto, ancho). El processor redimensiona y normaliza.")

    # -----------------------------------------------------------------
    # ENCODER: imagen -> secuencia de vectores
    # -----------------------------------------------------------------
    separador("3. ENCODER (Swin Transformer): imagen -> vectores")
    print("Qué hace: divide la imagen en 'patches', los proyecta a embeddings y")
    print("aplica atención con ventanas deslizantes (shifted windows).")
    with torch.no_grad():
        encoder_outputs = model.encoder(pixel_values)
    memoria = encoder_outputs.last_hidden_state
    print(f"\nSalida del encoder (memoria).shape = {tuple(memoria.shape)}")
    print("  -> (batch, N_patches, D). Es la SECUENCIA de vectores visuales.")
    print(f"  -> N = {memoria.shape[1]} vectores, cada uno de dimensión D = {memoria.shape[2]}.")
    print("Esta 'memoria' es lo que el decoder mirará vía cross-attention.")

    # -----------------------------------------------------------------
    # DECODER + CROSS-ATTENTION: un paso
    # -----------------------------------------------------------------
    separador("4. DECODER + CROSS-ATTENTION (un paso de generación)")
    print("El decoder es autoregresivo: genera un token a la vez.")
    print("En cada paso combina:")
    print("  (a) self-attention   -> mira los tokens ya generados")
    print("  (b) cross-attention  -> mira la 'memoria' del encoder (la imagen)")
    print("  (c) feed-forward     -> transforma la representación")

    start_token = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        start_token, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    print(f"\ndecoder_input_ids.shape = {tuple(decoder_input_ids.shape)}  (tokens iniciales)")

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )
    print(f"logits.shape = {tuple(out.logits.shape)}")
    print("  -> (batch, seq_len, vocab). Distribución sobre el vocabulario para el próximo token.")

    if out.cross_attentions is not None:
        ca = out.cross_attentions[0]
        print(f"\ncross_attention[capa 0].shape = {tuple(ca.shape)}")
        print("  -> (batch, n_heads, tokens_decoder, N_patches_encoder).")
        print("  Cada cabeza indica CUÁNTO atiende cada token de salida a cada patch de la imagen.")
        print("  ¡Esta es la conexión encoder<->decoder!")

    # -----------------------------------------------------------------
    # RESUMEN DEL FLUJO
    # -----------------------------------------------------------------
    separador("5. RESUMEN DEL FLUJO COMPLETO")
    print(
        """
    (1) Imagen                      (B, 3, H, W)
          |  processor: resize + normalize
          v
    (2) pixel_values                (B, 3, H', W')
          |  ENCODER Swin: patches -> embeddings -> shifted-window attention
          v
    (3) memoria (vectores imagen)   (B, N, D)
          |
          |   +----------------------------------------------+
          |   |            DECODER (autoregresivo)           |
          +-->|  self-attn(tokens) + CROSS-ATTN(memoria) +FFN|
              +----------------------------------------------+
                          |  (repetir token a token)
                          v
    (4) logits -> token -> ... -> <eos>
                          |
                          v
    (5) Secuencia de tokens  ->  Texto / JSON estructurado
    """
    )


if __name__ == "__main__":
    main()
