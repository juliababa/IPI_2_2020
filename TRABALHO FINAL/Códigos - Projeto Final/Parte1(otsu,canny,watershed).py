import cv2
import numpy as np

##############################################################################
# PRÉ TRATAMENTO #
#  Carrega a imagem
image = cv2.imread('ORIGINAL.jpg')

# Aplica os filtros 
median = cv2.medianBlur(image,11)
blur = cv2.GaussianBlur(median,(7,7),1)

# Converte para escalas de cinza
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Binariza a imagem pelo método de Otsu
ret, thresh =  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsu', thresh)

##############################################################################
# MÉTODO DE CANNY #

# Encontra as bordas
edged = cv2.Canny(thresh, 30, 120)

# Encontra os contornos
contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Desenha os contornos
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
cv2.imshow('Contours', image)
cv2.waitKey(0)

##############################################################################
# SEGMENTAÇÃO WATERSHED #

# Marcadores
fg = cv2.erode(thresh, None, iterations=2)
bgt = cv2.dilate(thresh, None, iterations=3)
ret, bg = cv2.threshold(bgt, 1,128,1)

marker = cv2.add(fg, bg)
marker32 = np.int32(marker)
cv2.imshow('marker', marker)

# Aplica a segmentação watershed
cv2.watershed(image, marker32)
m = cv2.convertScaleAbs(marker32)
cv2.imshow('m', m)


ret, thresh =  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(image,image, mask=thresh)
cv2.imshow('final', res)
cv2.imwrite('FINAL.jpg', res)
cv2.waitKey(0)