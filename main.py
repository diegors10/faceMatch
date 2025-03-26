import cv2
import mediapipe as mp

# Inicializa os módulos do MediaPipe para detecção de rostos e desenho
reconhecimento_rosto = mp.solutions.face_detection
desenho = mp.solutions.drawing_utils
reconhecedor_rosto = reconhecimento_rosto.FaceDetection()

# Inicializa a captura da webcam
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break

    # Processa a imagem para detectar rostos
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lista_rostos = reconhecedor_rosto.process(imagem)

    # Se houver detecções, desenha os rostos na imagem
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    # Exibe a imagem processada
    cv2.imshow("Rostos na sua webcam", frame)

    # Encerra o loop se a tecla ESC for pressionada
    if cv2.waitKey(5) == 27:
        break

# Libera a webcam e fecha as janelas
webcam.release()
cv2.destroyAllWindows()
