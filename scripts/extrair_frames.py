"""
extrair_frames.py — Extrai frames de um vídeo para anotação no Roboflow.

Uso:
    python scripts/extrair_frames.py --video caminho/do/video.mp4 --saida frames/ --intervalo 1.0

Argumentos:
    --video      Caminho para o arquivo de vídeo (MP4, AVI, MOV, etc.)
    --saida      Pasta onde os frames serão salvos (criada automaticamente)
    --intervalo  Intervalo em segundos entre frames (padrão: 1.0)
    --max        Número máximo de frames a extrair (padrão: sem limite)
    --upload     Se presente, faz upload automático para o Roboflow

Depois de extrair os frames:
1. Acesse https://app.roboflow.com
2. Abra o projeto bovinsights (ou crie um novo)
3. Clique em "Upload" e envie a pasta gerada
4. Anote os frames: marque os bovinos deitados como "lying"
5. Treine uma nova versão do modelo
"""

import argparse
import os
import sys
import cv2


def extrair_frames(video_path: str, saida: str, intervalo_s: float, max_frames: int | None):
    if not os.path.exists(video_path):
        print(f"[ERRO] Vídeo não encontrado: {video_path}")
        sys.exit(1)

    os.makedirs(saida, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao_s = total_frames / fps if fps > 0 else 0

    print(f"Vídeo: {video_path}")
    print(f"FPS: {fps:.1f} | Frames totais: {total_frames} | Duração: {duracao_s:.1f}s")
    print(f"Intervalo de extração: {intervalo_s}s → ~{int(duracao_s / intervalo_s)} frames esperados")
    print()

    frame_step = max(1, int(fps * intervalo_s))
    frame_idx = 0
    extraidos = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            nome = f"frame_{frame_idx:06d}.jpg"
            caminho = os.path.join(saida, nome)
            cv2.imwrite(caminho, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            extraidos += 1
            print(f"\r  Extraídos: {extraidos} frames", end="", flush=True)

            if max_frames and extraidos >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"\n\n[OK] {extraidos} frames salvos em: {saida}")
    print()
    print("Próximos passos:")
    print("  1. Acesse https://app.roboflow.com e abra seu projeto")
    print("  2. Clique em 'Upload Images' e selecione a pasta:", saida)
    print("  3. Anote os bovinos deitados com a classe 'lying'")
    print("  4. Treine uma nova versão e atualize ROBOFLOW_MODEL_ID no .env")


def upload_roboflow(saida: str):
    """Faz upload dos frames diretamente para o Roboflow via SDK."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERRO] roboflow não instalado. Rode: pip install roboflow")
        sys.exit(1)

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        # Tentar carregar do .env
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ROBOFLOW_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]

    if not api_key:
        print("[ERRO] ROBOFLOW_API_KEY não encontrada. Defina no .env ou como variável de ambiente.")
        sys.exit(1)

    workspace = input("Workspace do Roboflow (ex: minha-fazenda): ").strip()
    project = input("Nome do projeto (ex: bovinos-confinamento): ").strip()

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    arquivos = [f for f in os.listdir(saida) if f.endswith(".jpg")]
    print(f"\nFazendo upload de {len(arquivos)} imagens para {workspace}/{project}...")

    for i, nome in enumerate(arquivos, 1):
        caminho = os.path.join(saida, nome)
        proj.upload(caminho, batch_name="frames-video-confinamento", tag_names=["lying-candidates"])
        print(f"\r  Upload: {i}/{len(arquivos)}", end="", flush=True)

    print(f"\n[OK] Upload concluído! Acesse https://app.roboflow.com/{workspace}/{project} para anotar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai frames de vídeo para anotação no Roboflow")
    parser.add_argument("--video", required=True, help="Caminho para o vídeo")
    parser.add_argument("--saida", default="frames_extraidos", help="Pasta de saída")
    parser.add_argument("--intervalo", type=float, default=1.0, help="Intervalo em segundos entre frames")
    parser.add_argument("--max", type=int, default=None, dest="max_frames", help="Máximo de frames")
    parser.add_argument("--upload", action="store_true", help="Faz upload para o Roboflow após extração")
    args = parser.parse_args()

    extrair_frames(args.video, args.saida, args.intervalo, args.max_frames)

    if args.upload:
        upload_roboflow(args.saida)
