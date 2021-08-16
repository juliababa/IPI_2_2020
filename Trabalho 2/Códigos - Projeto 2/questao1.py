import cv2 as cv
import numpy as np 

# Carrega a imagem que vamos utilizar e o seu negativo
img = cv.imread("morf_test.png", cv.IMREAD_GRAYSCALE)
orig_n = cv.bitwise_not(img)

# Binariza a imagem diretamente 
ret,thresh1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Mostra a binarização pura sem tratamentos
cv.imshow('Original', img)
cv.imshow('Binarizacao', thresh1)
cv.waitKey(0)
cv.destroyAllWindows()

#Tranformação morfológica(top-hat)
kernel = np.ones((7,7), np.uint8)
abertura_n = cv.morphologyEx(orig_n, cv.MORPH_OPEN, kernel)
top_hat = cv.bitwise_not(cv.subtract(orig_n, abertura_n))

# Binariza a imagem após a transformação morfológica
ret,thresh2 = cv.threshold(top_hat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Mostra a imagem transformada e a binarizada após a transformação
cv.imshow('Top-hat', top_hat)
cv.imshow('Binarizacao 2', thresh2)
cv.waitKey(0)
cv.destroyAllWindows()
