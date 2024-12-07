import cv2 as cv
import functions
import os

# Diretório de entrada com imagens
input_dir = r"C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\detector de mascara\\detector-de-mascaras\\imagens\\imagens_analizar"

# Diretórios de saída para salvar as classificações
output_dir_with_mask = r"C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\detector de mascara\\detector-de-mascaras\\imagens\\imagens_processadas\\com_mascara"
output_dir_without_mask = r"C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\detector de mascara\\detector-de-mascaras\\imagens\\imagens_processadas\\sem_mascara"

# Criando os diretórios de saída, se não existirem
os.makedirs(output_dir_with_mask, exist_ok=True)
os.makedirs(output_dir_without_mask, exist_ok=True)

# Lista todas as imagens no diretório de entrada
image_files = [file for file in os.listdir(input_dir) if file.endswith(('.jpg', '.png', '.webp'))]

# Carregando o modelo e dados para classificação
dataframe = functions.load_dataframe()  # Carregando dataframe com as imagens para treinamento
X_train, y_train = functions.train_test(dataframe)  # Dividindo conjuntos de treino e teste
pca = functions.pca_model(X_train)  # Modelo PCA para extração de features da imagem
X_train = pca.transform(X_train)  # Conjunto de treino com features extraídas
knn = functions.knn(X_train, y_train)  # Treinando modelo classificatório KNN

# Rótulo das classificações
label = {
    0: "Sem mascara",
    1: "Com mascara"
}

# Loop para processar cada imagem
for i, file_name in enumerate(image_files):
    file_path = os.path.join(input_dir, file_name)

    # Lendo a imagem
    frame = cv.imread(file_path)
    if frame is None:
        print(f"Erro ao carregar a imagem: {file_name}")
        continue

    # Transformando a imagem em escala de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectando região de interesse (ROI) no centro da imagem
    height, width, _ = frame.shape
    pt1, pt2 = ((width // 2) - 100, (height // 2) - 100), ((width // 2) + 100, (height // 2) + 100)
    region = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    gray_face = cv.cvtColor(region, cv.COLOR_BGR2GRAY)

    # Redimensionando e extraindo features
    gray_face = cv.resize(gray_face, (160, 160))
    vector = pca.transform([gray_face.flatten()])  # Extraindo features da imagem

    # Realizando a classificação
    pred = knn.predict(vector)[0]
    classification = label[pred]

    # Definindo a cor do retângulo (visualização, opcional)
    color = (0, 255, 0) if pred == 1 else (0, 0, 255)

    # Escrevendo a classificação na imagem
    cv.putText(frame, classification, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)
    cv.rectangle(frame, pt1, pt2, color, thickness=3)

    # Salvando a imagem no diretório correspondente
    output_path = os.path.join(output_dir_with_mask if pred == 1 else output_dir_without_mask, file_name)
    cv.imwrite(output_path, frame)
    print(f"Imagem {i + 1}/{len(image_files)} processada e salva em: {output_path}")

print("Processamento concluído!")
