#Importando a função da questão 1.1
from dec_int import *

#Importando a função da quetão 1.2
from egde_improv import*

#Função para interpolação bicúbica
def bicubic_interpolation(img, N):
    #imagem dizimada
    img_resized =cv2.resize(img, (0,0), fx=1/N, fy=1/N, interpolation=cv2.INTER_CUBIC) 
    #imagem interpolada
    img_resized2 =cv2.resize(img_resized, (0,0), fx=N, fy=N, interpolation=cv2.INTER_CUBIC)

    return img_resized2

#Lendo a imagem test80.png
img_test80 = cv2.imread('test80.jpg')

#Questão 1.3 (parte I) - Interpolação pelo vizinho mais próximo da imagem 
#através do uso da função dec_int e exibição dos resultados
img_nearest_neighbor = dec_int(img_test80, 2)
cv2.imshow('Interpolacao pelo vizinho mais proximo', img_nearest_neighbor)
cv2.waitKey(0)

#Questão 1.3 (parte II) - Interpolação bicúbica da imagem 
#e exibição dos resultados
img_bicubic = bicubic_interpolation(img_test80, 2)
cv2.imshow('Interpolacao bicubica', img_bicubic)
cv2.waitKey(0)

#Questão 1.3 (parte III) - Melhoramento da qualidade das imagens interpoladas
#através do uso da função edge_improv e exibição dos resultados

improved_nn = egde_improv(img_nearest_neighbor)
cv2.imshow('Filtro aplicado a interpolacao pelo vizinho mais proximo', improved_nn)
cv2.imwrite('nn_filtrada.jpg', improved_nn)
cv2.waitKey(0)

improved_bicubic = egde_improv(img_bicubic)
cv2.imshow('Filtro aplicado a interpolacao bicubica', improved_bicubic)
cv2.imwrite('bc_filtrada.jpg', improved_bicubic)
cv2.waitKey(0)
