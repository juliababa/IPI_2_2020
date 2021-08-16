from PIL import Image

# Carrega a imagem gerada no processo anterior
im = Image.open('FINAL.jpg')

# Converte para níveis de cinza
im_grey = im.convert('LA')

# Armazena as dimensões da imagem
width, height = im.size

# Inicializa os contadores
n = 0
total = 0

# Percorre a imagem 
for i in range(0, width):
    for j in range(0, height):
        # Armazena o valor da intensidade do pixel
        k = im_grey.getpixel((i,j))[0]
        # Efetua as operações apenas para a ROI
        if k != 0:
            n += 1
            total += im_grey.getpixel((i,j))[0]

# Calcula a média das intensidade dos pixels
mean = total / (n)

# Mostra o resultado
print(mean)