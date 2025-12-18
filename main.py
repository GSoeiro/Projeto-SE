import time
import cv2 # Neste caso atual apenas usamos o OpenCV para o processamento do vídeo e o HOG+SVM para deteção de pessoas (disponibilizado no OpenCV)
import numpy as np
import os
from flask import Flask, Response
from notification import send_pushsafer_notification

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECT_WIDTH = 320
DETECT_HEIGHT = 240
PROCESS_EVERY_N_FRAMES = 3
frame_count = 0

#-------------------------------
# VALORES PARA DETEÇÃO DE QUEDA
#-------------------------------

DROP_THRESHOLD = 0.09 # percentagem da altura do frame que determina
                      # quando a pessoa "baixou" o suficiente para considerar o ínicio de uma queda
FALL_STILL_TIME = 2.0 # tempo de espera para assumir uma queda

center_baseline = None
fall_start_time = None
fall_detected = False

#------------------------
# NOTIFICAÇÃO & COOLDOWN
#------------------------

fall_notified = False
last_fall_notification = 0
COOLDOWN_SECONDS = 30

#------------------------------------------------
# LOGS NO WEBSITE & SCREENSHOTS PARA NOTIFICAÇÃO
#------------------------------------------------

if not os.path.exists("logs"):
    os.makedirs("logs")

if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

#--------------------
# FUNÇÕES AUXILIARES
#--------------------

def save_screenshot(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"screenshots/queda_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print("Screenshot guardado:", filename)
    return filename


def log_queda(screenshot_path):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - Queda detetada - {screenshot_path}\n"
    with open("logs/quedas.log", "a") as f:
        f.write(line)
    print("LOG gravado:", line)

#-------------------
# DETEÇÃO DA PESSOA
#-------------------

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # Utilizamos o HOG com o modelo SVM (classificador pré-treinado para detetar pessoas)

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

app = Flask(__name__)


USE_PICAMERA = True
try:
    from picamera2 import Picamera2
except:
    USE_PICAMERA = False

if USE_PICAMERA:
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    cam.configure(config)
    cam.start()
else:
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

time.sleep(0.5)
last_boxes = None

#------------
# LIVESTREAM
#------------

def generate_frames():
    global frame_count, last_boxes
    global center_baseline, fall_start_time, fall_detected
    global fall_notified, last_fall_notification

    while True:

        # CAPTURA DO FRAME
        if USE_PICAMERA:
            frame = cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cam.read()
            if not ret:
                continue

        frame = cv2.rotate(frame, cv2.ROTATE_180) # RODAMOS A CÂMARA 180 GRAUS DERIVADO À POSIÇÃO DELA NO RPI
        annotated = frame.copy()
        H, W = frame.shape[:2]    
        small = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT)) # REDUÇÃO DA RESOLUÇÃO PARA UMA DETEÇÃO MAIS RÁPIDA
        frame_count += 1

        #--------------------------------
        # PROCESSAMENTO DE N EM N FRAMES
        #--------------------------------

        # Executa a deteção apenas a cada N frames,
        # reduzindo o custo computacional
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:

            # Aplica o detetor HOG + SVM na imagem reduzida
            # winStride -> movimentação da "box"
            # padding   -> margem adicional para deteção
            # scale     -> fator de escala para diferentes tamanhos de pessoa
            rects, weights = hog.detectMultiScale(
                small, winStride=(4, 4), padding=(8, 8), scale=1.03
            )

            # Verifica se foi detetada pelo menos uma pessoa
            if len(rects) > 0:

                # Ordena as deteções pela área (largura x altura),
                # assumindo que a maior corresponde à pessoa principal
                rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)

                # Guarda a bounding box da pessoa principal
                last_boxes = rects[0]

            else:
                # Se não houver deteções, limpa a bounding box anterior
                last_boxes = None

        #-------------------
        # DETEÇÃO DA PESSOA
        #-------------------

                    # Verifica se existe uma deteção válida da pessoa
            if last_boxes is not None:

                # Obtém as coordenadas da bounding box
                # (x, y) -> canto superior esquerdo
                # (w, h) -> largura e altura
                x, y, w, h = last_boxes

                # Calcula os fatores de escala entre a imagem reduzida
                # (usada na deteção) e a imagem original
                scale_x = W / DETECT_WIDTH
                scale_y = H / DETECT_HEIGHT

                # Converte as coordenadas da bounding box
                # para o referencial da imagem original
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)

                # Filtra deteções demasiado pequenas,
                # reduzindo falsos positivos
                if w * h > 0.02 * W * H:

                    # Calcula o centro da bounding box
                    cx = x + w // 2
                    cy = y + h // 2

                    # Recalcula os limites da bounding box
                    # garantindo que ficam dentro da imagem
                    x1 = max(0, cx - w // 2)
                    y1 = max(0, cy - h // 2)
                    x2 = min(W - 1, cx + w // 2)
                    y2 = min(H - 1, cy + h // 2)

                    # Desenha a bounding box da pessoa
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Adiciona o rótulo "Pessoa" acima da bounding box
                    cv2.putText(annotated, "Pessoa", (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)

            #-------------    
            #BLUR DA FACE
            #-------------

            # Seleciona apenas a região superior da bounding box,
            # onde normalmente se encontra a face da pessoa
            face_area = annotated[y1:y1 + (y2 - y1)//3, x1:x2]

            # Converte a região da face para tons de cinzento,
            # necessário para o classificador Haar Cascade
            gray = cv2.cvtColor(face_area, cv2.COLOR_BGR2GRAY)

            # Deteção de faces utilizando Haar Cascade
            # 1.1 -> fator de escala
            # 5   -> número mínimo de vizinhos
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Para cada face detetada, aplica blur na região correspondente
            for (fx, fy, fw, fh) in faces:

                # Converte coordenadas da face para o referencial da imagem original
                fx1 = x1 + fx
                fy1 = y1 + fy
                fx2 = fx1 + fw
                fy2 = fy1 + fh

                # Aplica Gaussian Blur para "esconder" a face
                annotated[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(
                    annotated[fy1:fy2, fx1:fx2], (31, 31), 0
                )

                #----------------------------------------------
                # LÓGICA DA QUEDA (PELO PONTO CENTRAL NO CORPO)
                #----------------------------------------------

                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2

                if center_baseline is None:
                    center_baseline = center_y

                if abs(center_y - center_baseline) < H * 0.05: # 5% da altura da imagem 
                    center_baseline = 0.9 * center_baseline + 0.1 * center_y # 90% valor antigo + 10% valor novo

                low_position = center_y > center_baseline + H * DROP_THRESHOLD

                if low_position:
                    if fall_start_time is None:
                        fall_start_time = time.time()
                else:
                    if center_y < center_baseline + H * 0.10:
                        fall_start_time = None
                        fall_detected = False

                if fall_start_time is not None:
                    if time.time() - fall_start_time >= FALL_STILL_TIME:
                        fall_detected = True

                #-------------------------
                # PONTO CENTRAL NA IMAGEM
                #-------------------------

                color = (0,0,255) if fall_detected else (255,0,0)
                cv2.circle(annotated, (int(center_x), int(center_y)), 6, color, -1)

                #-----------------------------------------------------------
                # QUEDA DETETADA → screenshot, log, notificação (pushsafer)
                #-----------------------------------------------------------

                now = time.time()

                if fall_detected and (now - last_fall_notification > COOLDOWN_SECONDS):
                    screenshot_path = save_screenshot(annotated)
                    log_queda(screenshot_path)
                    send_pushsafer_notification(
                        "Queda Detetada!",
                        "Alerta de Queda",
                        image_path=screenshot_path
                    )
                    last_fall_notification = now
                    fall_notified = True

        else:
            center_baseline = None
            fall_start_time = None
            fall_detected = False

        if fall_detected:
            cv2.putText(annotated, "QUEDA DETETADA", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2)
        else:
            cv2.putText(annotated, "Estado: OK", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode(".jpg", annotated)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

# PÁGINA PRINCIPAL COM VÍDEO + LOGS

@app.route("/")
def index():

    try:
        with open("logs/quedas.log", "r") as f:
            logs = [line.strip() for line in f.readlines()][-12:]
    except:
        logs = ["Nenhuma queda registada."]

    log_html = "<br>".join([f"- {line}" for line in logs])

    html = f"""
    <h2>Deteção de Quedas</h2>

    <div style="display:flex; flex-direction:row; gap:40px;">

        <div>
            <img src="/video" width="640">
        </div>

        <div style="width:3px; background:#aaa;"></div>

        <div>
            <h2>LOGS:</h2>
            <div style="
                background:#f4f4f4;
                width:380px;
                height:460px;
                padding:15px;
                border-radius:10px;
                overflow-y:auto;
                border:1px solid #ccc;
                font-size:18px;">
                {log_html}
            </div>
        </div>

    </div>
    """
    return html


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("Abrir no browser: http://<IP_DO_PI>:5000/")
    app.run(host="0.0.0.0", port=5000, threaded=True)
