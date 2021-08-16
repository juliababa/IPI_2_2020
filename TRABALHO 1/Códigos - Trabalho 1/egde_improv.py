import cv2
import numpy as np

def egde_improv(img):
    #Convertendo para escala gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Aplicando o filtro de Gaussian
    blur = cv2.GaussianBlur(gray,(3,3),0)
    
    #Aplicar o operador Laplaciano em algum tipo de dados superior
    laplacian = cv2.Laplacian(blur,cv2.CV_64F)

    #Localizando as bordas pelo lado mais brilhante
    laplacian1 = laplacian/laplacian.max()
    
    return laplacian1