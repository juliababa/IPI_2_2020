import numpy as np
import cv2 as cv2
import math

#Questão 1.1 - interpolação por vizinho mais próximo, por um fator N(múltiplo de 2)

def dec_int(img, N):
    #extraindo o tamanho da imagem
    width,height,chan = img.shape
    
    #PARTE I - ENCONTRANDO A IMAGEM DIZIMADA
    #Calculando a larguras e alturas da imagem 
    w1 = int(width/N)
    h1 = int(height/N)

    #criando uma matriz para a imagem redimensionada 
    scaled_img = np.empty((w1, h1, chan), dtype=np.uint8)

    #calculando a escala
    x_ratio = float(width/float(w1))
    y_ratio = float(height/float(h1))

    #formando a imagem dizimada 
    for i in range(w1):
        for j in range(h1):
            p_x=math.floor(j*x_ratio)
            p_y=math.floor(i*y_ratio)

            scaled_img[i, j] = img[int(p_y), int(p_x)]
    
    #PARTE II - ENCONTRANDO A IMAGEM INTERPOLADA
    #copiando a imagem dizimada 
    copy_scaled_img = scaled_img.copy() 
    
    #extraindo as proporções da imagem dizimada
    w,h,c= copy_scaled_img.shape

    #calculando a largura e altura da imagem interpolada
    w2 = int(w*N)
    h2 = int(h*N)

    #criando uma matriz para a imagem redimensionada
    new_img = np.empty((w2, h2, c), dtype=np.uint8)

    #calculando a escala
    x_ratio1 = float(w/float(w2))
    y_ratio1 = float(h/float(h2))

    #formando a imagem interpolada 
    for i in range(w2):
        for j in range(h2):
            p_x=math.floor(j*x_ratio1)
            p_y=math.floor(i*y_ratio1)

            new_img[i, j] = copy_scaled_img[int(p_y), int(p_x)]
    
    return new_img
