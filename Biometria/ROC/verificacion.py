import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def get_data(dir_clien, dir_imp):
    clientes = open(dir_clien)
    impostores = open(dir_imp)

    scores =[]
    scoresCli =[]
    lineasClientes = clientes.readlines()
    for linea in lineasClientes:
        linea = linea.rstrip()
        #0 si es cliente
        scores.append([float(linea.split(" ")[1]),0])
        scoresCli.append(float(linea.split(" ")[1]))

    scoresImp=[]
    lineasImpostores = impostores.readlines()
    for linea in lineasImpostores:
        linea =linea.rstrip()
        #1 si es impostor
        scores.append([float(linea.split(" ")[1]),1])
        scoresImp.append(float(linea.split(" ")[1]))

    random.shuffle(scores)
    scores.sort(key = lambda x: x[0])
    scoresCli.sort(key = lambda x: x)
    scoresImp.sort(key = lambda x: x)
    return scores, scoresCli, scoresImp

def plot_roc(scores, scoresCli, scoresImp):
    nCli= len(scoresCli)
    nImp= len(scoresImp)
    xROC= []
    yROC= []
    xROC.append(1)
    yROC.append(1)
    impVistos=0
    cliVistos=0
    for i in range(0, len(scores)):
        if scores[i][1]==0:
            cliVistos+=1
        else:
            impVistos+=1
        
        FN = cliVistos/nCli 
        FP = (nImp - impVistos)/nImp 
        xROC.append(FP)
        yROC.append(1-FN)

    plt.plot(xROC,yROC)
    plt.title('curva ROC')
    plt.ylabel('1-FN')
    plt.xlabel('FP')
    plt.show()
    return xROC, yROC, scores

def ROC_area(scoresCli, scoresImp):
    nCli= len(scoresCli)
    nImp= len(scoresImp)
    ncli_mayor=0
    for cli in scoresCli:
        for imp in scoresImp:
            if cli>imp:
                ncli_mayor+=1
            elif cli==imp:
                ncli_mayor+=0.5
            else:
                break
    area = ncli_mayor/(nCli*nImp)
    return area

def d_prime(scoresCli, scoresImp):
  mCli = np.mean(scoresCli)
  mImp = np.mean(scoresImp)
  vCli = np.var(scoresCli)
  vImp = np.var(scoresImp)
  resultado = (mCli - mImp) / (math.sqrt(vCli + vImp))
  return resultado

def FN_given_FP_X(xRoc, yRoc, porcentaje, scores):
    indice, valor = min(enumerate(xRoc), key=lambda x: abs(x[1] - porcentaje))
    FN_X = 1-yRoc[indice]
    umbral= scores[indice][0]
    return umbral, FN_X

def FP_given_FN_X(xRoc, yRoc, porcentaje, scores):
    indice, valor = min(enumerate(yRoc), key=lambda x: abs(1-x[1] - porcentaje))
    FP_X = xRoc[indice]
    umbral= scores[indice][0]
    
    return umbral, FP_X

def FN_equal_FP(xRoc, yRoc, scores):
    v_1 = None
    v_2 = None
    umbral = None
    dif_min = float('inf')

    for i, FP_XROC in enumerate(xRoc):
        dif = abs(FP_XROC - (1-yRoc[i]))
        if dif < dif_min:
            dif_min = dif
            v_1 = FP_XROC
            v_2 = 1-yRoc[i]
            umbral=scores[i][0]
    
    return v_1, v_2, umbral


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Curva ROC verificacion')
    parser.add_argument('--clientes_dir', type=str, required=True, help='directorio del fichero de clientes')
    parser.add_argument('--impostores_dir', type=str, required=True, help='directorio del fichero de impostores')
    parser.add_argument('--x_FP', type=float, required=False, default=61, help='porcentaje de FP para calcular el de FN')
    parser.add_argument('--x_FN', type=float, required=False, default=1, help='porcentaje de FN para calcular el de FP')

    args = parser.parse_args()

    scores, scoresCli, scoresImp  = get_data(args.clientes_dir, args.impostores_dir)
    xRoc, yRoc, scores = plot_roc(scores, scoresCli, scoresImp)
    area = ROC_area(scoresCli, scoresImp)
    print("Area debajo de la curva ROC: " + str(area))
    dPrime=d_prime(scoresCli, scoresImp)
    print("Valor del dPrime: " + str(dPrime))
    umbral, FP_X = FP_given_FN_X(xRoc, yRoc, args.x_FP, scores)
    print("Para FN de " + str(args.x_FP) + " dado -> umbral: " + str(umbral) + ", X de FP: " +  str(FP_X))
    umbral, FN_X = FN_given_FP_X(xRoc, yRoc, args.x_FN, scores)
    print("Para FP de " + str(args.x_FN) + " dado -> umbral: " + str(umbral) + ", X de FN: " +  str(FN_X))
    FP_X, FN_X, umbral = FN_equal_FP(xRoc, yRoc, scores)
    print("Para el dataset de clientes e impostores dado FP = FN cuando -> umbral: " + str(umbral) + ", X de FP: " +  str(FP_X) + ", X de FN: " +  str(FN_X))




