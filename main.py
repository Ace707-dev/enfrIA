from ultralytics import YOLO
from pathlib import Path
import torch
import shutil
import pandas as pd
import gc
import cv2
import os
import time
import re

def main():
    # Configuración del dispositivo
    cuda_available = torch.cuda.is_available()
    print("CUDA disponible:", cuda_available)
    device = "cuda" if cuda_available else "cpu"
    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        gc.collect()

    # Configuración flexible
    MODEL_PATH = os.getenv("MODEL_PATH", "runs/healthy-rotten10/weights/best.pt")
    DATA_DIR = os.getenv("DATA_DIR", "archive/Fruit And Vegetable Diseases Dataset_sorted")

    ROOT = Path(__file__).resolve().parent
    DATA_DIR_PATH = ROOT / DATA_DIR
    HEALTHY_DIR = DATA_DIR_PATH / "healthy"
    ROTTEN_DIR = DATA_DIR_PATH / "rotten"

    HEALTHY_DIR.mkdir(exist_ok=True, parents=True)
    ROTTEN_DIR.mkdir(exist_ok=True, parents=True)

    model = YOLO(ROOT / MODEL_PATH)

    if cuda_available:
        model.to("cuda")

    # Configuración de la cámara (Windows a veces prefiere CAP_DSHOW)
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video.isOpened():
        print("ERROR: No se pudo acceder a la cámara.")
        return

    # Preparar directorio de capturas y contador para evitar sobrescribir
    captures_dir = ROOT / "capturas"
    captures_dir.mkdir(exist_ok=True)
    existing = os.listdir(str(captures_dir))
    max_idx = -1
    for f in existing:
        m = re.search(r"captura_(\d+)\.png$", f)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    counter = max_idx + 1

    interval = 5.0  # segundos entre capturas automáticas
    last_capture = time.time()

    log = []
    print("Presiona 'c' para capturar y clasificar, 'q' para salir. Captura automática cada", interval, "segundos.")

    csv_path = ROOT / "prediction_log.csv"

    while True:
        ret, frame = video.read()
        if not ret:
            print("ERROR: Falló la lectura de la cámara.")
            break

        # Mostrar video en vivo
        cv2.imshow("Camera", frame)

        now = time.time()
        take_auto = (now - last_capture) >= interval
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        take_manual = key == ord('c')

        if take_auto or take_manual:
            filename = captures_dir / f"captura_{counter:03d}.png"
            cv2.imwrite(str(filename), frame)
            print(("Auto" if take_auto else "Manual"), "captura:", filename.name)

            # Predecir directamente en el frame para mayor velocidad
            results = None
            try:
                results = model.predict(frame, device=device, imgsz=224, verbose=False)
                res = results[0]
                pred_class = res.probs.top1
                class_name = res.names[pred_class]
                confidence = float(res.probs.top1conf.item())

                # Retroalimentación visual en la ventana
                label = f"{class_name} {confidence:.1%}"
                color = (0, 200, 0) if 'healthy' in class_name.lower() else (0, 0, 200)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.imshow("Camera", frame)

                # Guardar una copia en el destino healthy/rotten
                target_dir = HEALTHY_DIR if "healthy" in class_name.lower() else ROTTEN_DIR
                shutil.copy(str(filename), str(target_dir / filename.name))

                # Registrar en memoria y agregar al CSV
                log.append((filename.name, class_name, confidence))
                df_row = pd.DataFrame([(filename.name, class_name, confidence)], columns=["image", "prediction", "confidence"])
                if not csv_path.exists():
                    df_row.to_csv(csv_path, index=False)
                else:
                    df_row.to_csv(csv_path, index=False, mode='a', header=False)

                print(f"Predicción: {class_name} ({confidence:.1%}) -> copia guardada en {target_dir.name}")
            except Exception as e:
                print("Falló la predicción:", e)
            finally:
                # Limpieza
                if results is not None:
                    del results
                if cuda_available:
                    torch.cuda.empty_cache()
                gc.collect()

            counter += 1
            last_capture = now

    video.release()
    cv2.destroyAllWindows()

    # Estadísticas resumen
    if log:
        healthy_count = sum(1 for _, pred, _ in log if "healthy" in pred.lower())
        rotten_count = len(log) - healthy_count
        avg_conf = sum(conf for _, _, conf in log) / len(log)
        print("\n" + "=" * 60)
        print(f"✓ Se capturaron y clasificaron {len(log)} imagen(es)")
        print(f"  Healthy: {healthy_count}")
        print(f"  Rotten:  {rotten_count}")
        print(f"  Confianza promedio: {avg_conf:.2%}")
        print("=" * 60)




if __name__ == "__main__":
    main()
