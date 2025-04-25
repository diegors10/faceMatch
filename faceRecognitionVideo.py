import cv2
import face_recognition
import numpy as np
import os

# Carregar imagens conhecidas e codificar os rostos
known_face_encodings = []
known_face_names = []

# Lista de imagens e nomes
known_images = [
    ("faces/1-face.png", "Ana"),
    ("faces/2-face.png", "Gabryel"),
    ("faces/3-face.png", "Karoline"),
    ("faces/4-face.png", "Rodrigo"),
    ("faces/10-face.png", "Diego"),    
]

print("Carregando imagens conhecidas...")

for image_path, name in known_images:
    if not os.path.exists(image_path):
        print(f"[AVISO] Imagem não encontrada: {image_path}")
        continue

    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)
        print(f"[INFO] Rosto codificado com sucesso: {name}")
    else:
        print(f"[ERRO] Nenhum rosto encontrado em {image_path}")

if not known_face_encodings:
    print("[FALHA] Nenhum rosto conhecido foi carregado.")
    exit()

# Inicializa a webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("[ERRO] Não foi possível acessar a webcam.")
    exit()

print("Reconhecimento facial iniciado. Pressione ESC para sair.")

while True:
    success, frame = webcam.read()
    if not success:
        print("[ERRO] Não foi possível ler o frame da webcam.")
        break

    # Reduz o tamanho do frame para acelerar o processamento (opcional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detecta os rostos e codifica
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Desconhecido"

        if face_distances.any():
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Redimensiona coordenadas de volta para o tamanho original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha retângulo ao redor do rosto
        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Exibe o resultado
    cv2.imshow("Reconhecimento Facial", frame)

    # Tecla ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
