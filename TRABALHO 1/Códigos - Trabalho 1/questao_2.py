import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

#Função para a realização da transformação gamma
def power_law(img, gamma):
  gamma_image = np.array(255*(img/255)**gamma, dtype = np.uint8)
  return gamma_image

#Função para equalização das imagens
def equalize(img):
  equalized_image = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
  return equalized_image

#Leitura das imagens
car = cv2.imread('car.png')
crowd = cv2.imread('crowd.png')
university = cv2.imread('university.png')

#Questão 2.1 - Correção gamma para cada imagem
#Sendo mostrado na tela apenas o melhor resultado 

gamma_car = power_law(car, 2.0)
cv2.imshow("Correcao Gamma - car.png", gamma_car)
cv2.waitKey(0)

gamma_crowd = power_law(crowd, 0.4)
cv2.imshow("Correcao Gamma - crowd.png", gamma_crowd)
cv2.waitKey(0)

gamma_university = power_law(university, 0.25)
cv2.imshow("Correcao Gamma - university.png", gamma_university)
cv2.waitKey(0)

#Questão 2.2 (parte I)- Equalização das três imagens 
#E exibição dos resultados 

equalized_car = equalize(car)
cv2.imshow('Equalizacao Crowd', equalized_car)
cv2.waitKey(0)

equalized_crowd = equalize(crowd)
cv2.imshow('Equalizacao Crowd' , equalized_crowd)
cv2.waitKey(0)

equalized_university = equalize(university)
cv2.imshow('Equalizacao University', equalized_university)
cv2.waitKey(0)

#Questão 2.2 (parte II) - Histograma e 
#CDF(função de distribuição acumulada) antes e depois da equalização
#da imagem university.png

#Histograma antes da equalização
university_hist = cv2.calcHist([university],[0],None,[256],[0,256]) 
plt.subplots(num = 'Histograma-ANTES')
plt.plot(university_hist)
plt.show()

#Histograma depois da equalização
university_hist_eq = cv2.calcHist([equalized_university],[0],None,[256],[0,256]) 
plt.subplots(num = 'Histograma-DEPOIS')
plt.plot(university_hist_eq)
plt.show()

#CDF antes da equalização
university_cdf = university_hist.cumsum()
plt.subplots(num = 'CDF-ANTES')
plt.plot(university_cdf)
plt.show()

#CDF depois da equalização
university_cdf_eq = university_hist_eq.cumsum()
plt.subplots(num = 'CDF-DEPOIS')
plt.plot(university_cdf_eq)
plt.show()







