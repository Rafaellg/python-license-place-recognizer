# -*- coding: utf-8 -*-
# Autores: Rafael Guimaraes e Taina Viriato

import cv2
import os

import DetectChars
import DetectPlates

# Variaveis de level
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

# Exibe passo a passo o processo
showSteps = False


def main():
    # Recupera valores do treinamento de KNN
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    # Verifica se o treinamento falhou
    if not blnKNNTrainingSuccessful:
        print "\nErro: Ocorreu um erro no treinamento de KNN\n"
        return

    # Abre a imagem
    imgOriginalScene = cv2.imread("images/placas07.jpg")

    # Verifica se a imagem original foi encontrada
    if imgOriginalScene is None:
        print "\nErro: Não foi possível ler a imagem de entrada \n\n"
        os.system("pause")
        return

    # Detecta placas
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

    # Detecta caracteres na placa
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    # Mostra imagem original
    cv2.imshow("imgOriginalScene", imgOriginalScene)

    # Verifica se foram encontradas placas
    if len(listOfPossiblePlates) == 0:
        print "\nNenhuma placa encontrada\n"
    else:
        # Ordena a lista de possiveis placas em ordem decrescente (Mais caracteres para menos caracteres)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # Inicia com a primeira placa
        licPlate = listOfPossiblePlates[0]

        # Mostra a placa e sua versao binarisada
        cv2.imshow("imgPlate", licPlate.imgPlate)
        #cv2.imshow("imgThresh", licPlate.imgThresh)

        # Verifica se existem caracteres nas placas
        if len(licPlate.strChars) == 0:
            print "\nNão foram encontrados caracteres\n\n"
            return

        # Desenha um retangulo ao redor da placa
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        # Printa o texto da placa
        print "\nPlaca lida da imagem = " + licPlate.strChars + "\n"
        print "----------------------------------------"

        # Escreve texto da placa na imagem
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        # Exibe a imagem original novamente
        cv2.imshow("imgOriginalScene", imgOriginalScene)

        # Escreve a imagem num arquivo de saida
        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

        # Espera interacao do usuario
        cv2.waitKey(0)

    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    # Recupera 4 vertices do retangulo rotacionado
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    # Aumenta o retangulo
    p2fRectPoints[0][1] += 10
    p2fRectPoints[1][1] -= 15
    p2fRectPoints[2][1] -= 15
    p2fRectPoints[3][1] += 10

    # Desenha 4 linhas vermelhas
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_GREEN, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    # Area central do texto
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    # Area esquerda do texto
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    # Seleciona a font do texto
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    # Seleciona a escala do texto
    fltFontScale = float(plateHeight) / 30.0
    # Seleciona a largura do texto
    intFontThickness = int(round(fltFontScale * 2.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,intFontThickness)

    # Desempacote o retangulo rotacionado no ponto central, largura, altura e angulo
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    # Certifica que o centro é inteiro
    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    # Area horizontal do texto igual a da placa
    ptCenterOfTextAreaX = int(intPlateCenterX)

    # Verifica se a placa é acima de 3/4 da imagem
    if intPlateCenterY < (sceneHeight * 0.75):
        # Guarda a posicao abaixo da placa para o texto
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        # Guarda a posicao abaixo da placa para o texto
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    # Recupera a largura e altura do texto
    textSizeWidth, textSizeHeight = textSize

    # Calcula a origem do inferior esquerdo da area do texto baseado no centro, largura e altura
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    # Escreve o texto na imagem
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_GREEN, intFontThickness)

if __name__ == "__main__":
    main()
