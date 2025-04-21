import cv2
import face_recognition
import numpy as np

# Carregar imagens conhecidas e codificar os rostos
known_face_encodings = []
known_face_names = []

# Adicione rostos conhecidos aqui (substitua pelos seus arquivos de imagem)
known_images = [
    ("faces/1-face.png", "Daniel"),
    ("faces/2-face.png", "Pablo"),
    ("faces/6-face.png", "Jessica"),
    ("faces/4-face.png", "Natalia"),
    ("faces/7-face.png", "Matheus"),
    ("faces/5-face.png", "Willer")
]


for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)

# Carregar a foto que você quer analisar
photo_path = "img_comp/TURMA_BRAVO_1.jpg"  # Substitua pelo caminho da sua foto
photo = face_recognition.load_image_file(photo_path)

# Redimensiona a foto para melhorar o desempenho (opcional)
photo = cv2.resize(photo, (0, 0), fx=0.5, fy=0.5)  # Reduz a foto pela metade

# Converte a foto para RGB (necessário para face_recognition)
rgb_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

# Detecta rostos e landmarks na foto
#face_locations = face_recognition.face_locations(rgb_photo, model="cnn")
face_locations = face_recognition.face_locations(rgb_photo) #HOG (Histogram of Oriented Gradients - Histograma de Gradientes Orientados)
face_encodings = face_recognition.face_encodings(rgb_photo, face_locations)

# Itera sobre cada rosto detectado na foto
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
        box_color = (255, 0, 0)  # Vermelho para rostos desconhecidos (BGR)


     # Aumenta o tamanho da caixa para não cobrir o rosto
    padding = 20  # Ajuste o valor conforme necessário
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(photo.shape[0], bottom + padding)
    right = min(photo.shape[1], right + padding)


    # Desenha uma caixa ao redor do rosto
    cv2.rectangle(photo, (left, top), (right, bottom), box_color, 2)

    # Exibe o nome abaixo da caixa do rosto
    cv2.rectangle(photo, (left, bottom - 30), (right, bottom), box_color, cv2.FILLED)
    cv2.putText(photo, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Salva o resultado em um arquivo de imagem
output_path = "resultado.jpg"
cv2.imwrite(output_path, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))  # Converte de volta para BGR antes de salvar
print(f"Resultado salvo em: {output_path}")

# Mostra a foto com os rostos detectados
cv2.imshow("Reconhecimento Facial em Foto", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()