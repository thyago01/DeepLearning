import numpy as np
from skimage import io, viewer, color
import matplotlib.pyplot as plt
import math

#####Camada convolucional: Wellton Thyago de Souza - 11325715

"""
    Função que realiza a extensão por zero, recebemos o tamanho do filtro e 
    com base nele ampliamos o tamanho da imagem para posteriores operações
    """

def Extensao_zero(image, kernel):
    new_d = ((kernel//2)+1)
    image_padded = np.zeros((image.shape[0]+new_d, image.shape[1]+new_d, image.shape[2]))
    image_padded[new_d-1:-new_d+1,new_d-1:-new_d+1] = image
    return np.int16(image_padded) 

"""
    Na convolução recebemos como parâmetros os valores da imagem, filtros e stride
    É realizado o calculo das dimensões que a imagem deverá possuir após a operação de convolução
    tendo sempre como base o tamanho do núcle e o stride. A variavel sub é a responsável por "armazenar"
    a área onde será feita a operação. 
    """
def Convolucao(image, k, stride, padding):
    k = np.flipud(np.fliplr(k))
    x_row, x_col = image.shape[0], image.shape[1]
    k_row, k_col = k.shape[0], k.shape[1]
    out_row, out_col = (x_row - k_row)//stride + 1, (x_col - k_col)//stride + 1
    out = np.int16(np.zeros((out_row, out_col)))
    for x in range(k_row//2, out_row,stride):
        for y in range(k_col//2, out_col,stride):
            sub = (image[x : x + k_row, y : y + k_col])
            out[x//stride - k_row//2,y//stride - k_col//2] = np.sum(sub * k)
            
    return out

def func_ativacao(func_type, z):
    if func_type == 'relu':
        return z * (z > 0)
    elif func_type == 'tanh':
        z = (2/(1 + np.exp(-2*z)))-1
        return z

"""
    Visualizar_fmap(Mapa de características)
    Função que convertendo-o para uma imagem em níveis de cinza no intervalo [0, 255], 
    com a) níveis em valor absoluto com preto = zero; e b) mínimo = preto.
    """
def Visualizar_fmap(image):
    preto_abs = np.abs(image)
    preto_abs -= preto_abs.min()
    preto_abs *= 255 / preto_abs.max()
    for camada in range(image.shape[2]):
        plt.imshow(preto_abs[:,:,camada], cmap='gray') 
        plt.show()

    preto_min = image
    preto_min -= preto_min.min()
    preto_min *= 255 / preto_min.max()
    
    for layer in range(image.shape[2]):
        plt.imshow(preto_min[:,:,layer], cmap='gray') 
        plt.show()

"""
    Implementa a camada convolucional utilizando bias(definido no arquivo csv) e função de ativação.
    """
    
def Camada_Convolucional(image,C,act_func, filters, stride, padding):
    h_x = image.shape[0]
    w_x = image.shape[1]
    filter_size = int(math.sqrt(filters.shape[1] - 1))
    h_filter = filter_size
    w_filter = filter_size
    h_out = int((h_x - h_filter + 2 * padding) / stride) + 1
    w_out = int((w_x - w_filter + 2 * padding) / stride) + 1
    fmap = np.zeros((h_out, w_out, C))
    filtro = np.zeros((h_filter, w_filter, image.shape[2]))
    linha = 0
    for fmap_camada in range(C):
        for camada in range(image.shape[2]):
            filtro[:,:,camada] = filters[linha,1:].reshape((filter_size, filter_size))
        bias = filters[linha,0]
        if(padding==1):
            fmap[1:-1,1:-1,fmap_camada] = func_ativacao(act_func, Convolucao(image,filtro, stride, padding) + bias)
        else:
            fmap[:,:,fmap_camada] = func_ativacao(act_func, Convolucao(image,filtro, stride, padding) + bias)
        linha+=1
    return fmap

"""
    Instruções para uso
    Carregamos a imagem a ser trabalhada
    Carregamos os filtros e bias
    Definimos stride e padding
    Calculo de convoluções
    Exibição do mapa de características
    """

img = io.imread('ImagensDeTeste/IMG_0103.png')
convFilters = np.loadtxt('filtros.csv', delimiter=',', dtype=float, skiprows=1)
img = color.gray2rgb(img)
filter_size = int(math.sqrt(convFilters.shape[1] - 1))
teste = img
stride = int(input("Entre com um valor inteiro positivo para o stride(Casos onde o stride é maior que um e exite a extensão não foram tratados. Aumente o valor de stride apenas se não for usar extensão por zero) : "))
padding = int(input("Sua operaçao será feita com padding? 1(Sim) ou 0(Não): "))
if(padding==1):
    teste = Extensao_zero(img,filter_size)
teste = Camada_Convolucional(teste,convFilters.shape[0],'relu',convFilters,stride,padding)
Visualizar_fmap(teste)
