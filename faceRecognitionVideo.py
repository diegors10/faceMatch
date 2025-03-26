import cv2
import face_recognition
import numpy as np

# Carregar imagens conhecidas e codificar os rostos
known_face_encodings = []
known_face_names = []

# Adicione rostos conhecidos aqui (substitua pelos seus arquivos de imagem)
known_images = [
    ("faces/pessoa1.jpg", "Diego"),
    ("faces/pessoa2.jpg", "Fernando"),
    ("faces/pessoa3.jpg", "Carlos")
]

for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)

# Inicializa a captura da webcam
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, frame = webcam.read()
    if not success:
        break

    # Converte para RGB (necessário para face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta rostos e landmarks usando dlib (via face_recognition)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Itera sobre cada rosto detectado
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Compara o rosto detectado com os rostos conhecidos
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Calcula a distância entre o rosto detectado e os rostos conhecidos
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        
        # Se houver uma correspondência, usa o nome correspondente
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

          # Define a cor da caixa com base no nome
        if name != "Desconhecido":
            box_color = (0, 255, 0)  # Verde para rostos conhecidos (BGR)
        else:
            box_color = (0, 0, 255)  # Vermelho para rostos desconhecidos (BGR)


        # Aumenta o tamanho da caixa para não cobrir o rosto
        padding = 20  # Ajuste o valor conforme necessário
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)


        # Desenha uma caixa ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Exibe o nome abaixo da caixa do rosto
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  

        # # Desenha uma caixa ao redor do rosto
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # # Exibe o nome abaixo da caixa do rosto
        # cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
        # cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostra a imagem com os rostos detectados
    cv2.imshow("Reconhecimento Facial", frame)

    # Sai do loop se a tecla ESC (27) for pressionada
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Libera a webcam e fecha a janela
webcam.release()
cv2.destroyAllWindows()