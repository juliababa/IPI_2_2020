import cv2 as cv
import numpy as np

# Lê a imagem cookies.tif.
img = cv.imread("cookies.tif", cv.IMREAD_GRAYSCALE)

# Binariza a imagem de acordo com um threshold determinado experimentalmente.
ret, threshold1 = cv.threshold(img, 55, 255, cv.THRESH_BINARY)

# Mostra a imagem original e sua versão binarizada
cv.imshow('original', img)
cv.imshow('binarizada', threshold1)
cv.waitKey(0)
cv.destroyAllWindows()

# Realiza operacoes morfologicas de abertura e fechamento pra remover elementos indesejados.
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(threshold1, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

# Mostra a imagem após as transformações morfológicas
cv.imshow('opening_closing', closing)
cv.waitKey(0)
cv.destroyAllWindows()

# Transformada hit-or-miss
mini = closing[8:161, 9:156]		# Obtem formato da regiao que queremos pra usar de mascara.
erode = cv.erode(closing, mini)
dilate = cv.dilate(erode, ~mini)
hit_or_miss = cv.subtract(erode, dilate)

# Remoção da cooky mordida
removed = cv.dilate(hit_or_miss, mini)

# Mostra a imagem após a remoção da cooky mordida
cv.imshow('removida', removed)
cv.waitKey(0)
cv.destroyAllWindows()

# Obtenção da imagem final em niveis de cinza com somente a cooky completa
final_img = img & removed

# Mostra a imagem final com apenas a cooky completa
cv.imshow('imagem final', final_img)
cv.waitKey(0)
cv.destroyAllWindows()


