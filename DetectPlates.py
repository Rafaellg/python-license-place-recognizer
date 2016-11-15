# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# Variaveis de modulo
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    # Verifica se esta no modo debug
    if Main.showSteps:
        cv2.imshow("0", imgOriginalScene)

    # Executa o pré-processamento para recuperar as imagens de escala de cinza e binária
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)

    if Main.showSteps:
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)

    # Encontra todos os possiveis caracteres na cena
    # Esta funcao primeiramente encontra todos os contornos e inclui apenas os contornos que podem ser um caracter (sem comparar por enquanto)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.showSteps:
        print "step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene))         # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if

    # A partir da lista de todos os possiveis caracteres, encontre grupos de caracteres semelhantes
    # No proximo passo cada grupo tentará ser reconhecido como uma placa
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps:
        print "step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(len(listOfListsOfMatchingCharsInScene))

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if

    # Itera sobre cada grupo de caracteres
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        # Tenta reconhecer a placa
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)

        # Verifica se foi encontrada
        if possiblePlate.imgPlate is not None:
            # Adiciona na lista de possiveis placas
            listOfPossiblePlates.append(possiblePlate)
        # end if
    # end for

    print "\n" + str(len(listOfPossiblePlates)) + " possíveis placas encontradas"

    if Main.showSteps:
        print "\n"
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_GREEN, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_GREEN, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_GREEN, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_GREEN, 2)

            cv2.imshow("4a", imgContours)

            print "possível placa " + str(i) + ". Clique em qualquer imagem e pressione uma tecla para continuar."

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print "\nDetecção de placa completada. Clique on qualquer imagem e pressione uma tecla para iniciar o reconhecimento de caracteres.\n"
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function


def findPossibleCharsInScene(imgThresh):
    # Valor a ser retornado
    listOfPossibleChars = []

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        if Main.showSteps:
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
        # end if

        possibleChar = PossibleChar.PossibleChar(contours[i])

        # Verifica se o contorno é um possível caracter
        if DetectChars.checkIfPossibleChar(possibleChar):
            # Incrementa contado
            intCountOfPossibleChars += 1
            # Adiciona na lista
            listOfPossibleChars.append(possibleChar)
        # end if
    # end for

    if Main.showSteps:
        print "\nstep 2 - len(contours) = " + str(len(contours))
        print "step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars)
        cv2.imshow("2a", imgContours)
    # end if

    return listOfPossibleChars
# end function


def extractPlate(imgOriginal, listOfMatchingChars):
    # Valor retornado
    possiblePlate = PossiblePlate.PossiblePlate()

    # Ordena os caracteres da esquerda pra direita baseado na posição X
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    # Calcula o centro da placa
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # Calcula a altura e largura da placa
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # Calcula o angulo correto da reguao da placa
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # Reune o ponto central da regiao da placa, largura, altura e o angulo de rotacao em um retangulo rotacionado membro da variavel da placa
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # Executa a rotacao atual

    # Recupera a matriz de rotacao para o angulo encontrado
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    # Recupera os valores originais da imagem
    height, width, numChannels = imgOriginal.shape

    # Rotaciona a imagem inteira
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    # Guarda a imagem cortada da placa
    possiblePlate.imgPlate = imgCropped

    return possiblePlate
# end function