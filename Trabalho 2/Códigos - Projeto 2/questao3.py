import cv2 as cv
import numpy as np

# Lê a imagem img_cells.jpg
img = cv.imread("img_cells.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Binariza a imagem diretamente 
ret,thresh1 = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Mostra a imagem original e sua versão binarizada
cv.imshow('original', img)
cv.imshow('binarizada', thresh1)
cv.waitKey(0)
cv.destroyAllWindows()

# Preenche os buracos
thresh1_neg = cv.bitwise_not(thresh1)
contour = cv.findContours(thresh1_neg, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)[0]
for i in contour:
    cv.drawContours(thresh1_neg, [i], 0, 255, -1)

# Remoção de ruídos na imagem
kernel = np.ones((4,4), np.uint8)
opening = cv.morphologyEx(thresh1_neg, cv.MORPH_OPEN, kernel)

# Mostra a imagem com os buracos preenchidos e tratamento de ruídos
cv.imshow("preenchimento", thresh1_neg)
cv.imshow("tratamento de ruidos", opening)
cv.waitKey(0)
cv.destroyAllWindows()

# Encontrando a área desconhecida
backgound = cv.dilate(opening, kernel,iterations=3)
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
foreground = cv.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)[1]
foreground = np.uint8(foreground)
unknown = cv.subtract(backgound, foreground)

# Mostra a imagem da área desconhecida
cv.imshow("area desconhecida", unknown)
cv.waitKey(0)
cv.destroyAllWindows()

# Rotulagem de marcador
markers = cv.connectedComponents(foreground)[1]

# Fazemos isso para que o fundo seja 1
markers = markers+1

# Marcando a região desconhecida com zero
markers[unknown==255] = 0

#Segmentação Watersheed
markers = cv.watershed(img, markers)
img[markers == -1] = [255,0,0]

# Mostra a o resultado final da segmentação de Watersheed
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()