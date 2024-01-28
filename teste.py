import cv2
import numpy as np

def resize_images(image1, image2):
    # Obtém as dimensões das imagens
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    
    # Redimensiona a imagem 2 para as dimensões da imagem 1
    if height1 != height2 or width1 != width2:
        image2 = cv2.resize(image2, (width1, height1))
    
    return image1, image2

def compare_images(image1_path, image2_path):
    # Carrega as imagens
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Redimensiona as imagens se necessário
    image1, image2 = resize_images(image1, image2)

    # Calcula a diferença entre as duas imagens
    diff = cv2.subtract(image1, image2)
    b, g, r = cv2.split(diff)

    # Verifica se todos os canais de cor são zero (ou seja, os pixels são iguais)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    return False

def extract_pokemon_sprite(image_path, template_path):
    # Carrega a captura de tela e o template do sprite do Pokémon
    screenshot = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    # Inicializa o detector ORB
    orb = cv2.ORB_create()

    # Encontra os pontos-chave e descritores na captura de tela e no template
    kp1, des1 = orb.detectAndCompute(screenshot, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # Verifica se os descritores foram encontrados para o template
    if des2 is None:
        print("Não foi possível encontrar descritores para o template.")
        return

    # Converte os descritores para o tipo de dados correto
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # Inicializa o matcher BF (Brute-Force)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Faz a correspondência dos descritores
    matches = bf.match(des1, des2)

    # Ordena as correspondências com base na distância
    matches = sorted(matches, key=lambda x: x.distance)

    # Extrai o sprite do Pokémon
    good_matches = matches[:10]  # Usando apenas as 10 melhores correspondências
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcula a matriz de transformação
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Aplica a matriz de transformação para encontrar a região do sprite do Pokémon na captura de tela
    h, w = template.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    x_min, y_min = np.int32(dst.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(dst.max(axis=0).ravel() + 0.5)
    pokemon_sprite = screenshot[y_min:y_max, x_min:x_max]

    # Salva o sprite do Pokémon como uma imagem separada em PNG
    cv2.imwrite("pokemon_sprite.png", pokemon_sprite)

    print("Sprite do Pokémon extraído com sucesso!")

# Chama a função para extrair o sprite do Pokémon
extract_pokemon_sprite('hq720.jpg', 'charizard.png')

# Caminhos das imagens
image2_path = "hq720.jpg"
image1_path = "charizard.png"

# Comparação das imagens
if compare_images(image1_path, image2_path):
    print("As imagens são iguais.")
else:
    print("As imagens são diferentes.")
