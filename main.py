import os

#import the lib


import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

# First thing is to capture the video

cap = cv2.VideoCapture(0)
Classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread("Resources/arrow.png", cv2.IMREAD_UNCHANGED)
classIDBin = 0

#Import all the Waste Images
imgWasteList = []         #list will contain all the 8 images
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste) #variable
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

  
# 0 Recyclable
# 1 Hazardous
# 2 Food Waste
# 3 Residual Waste

classDic = {0:None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}


#Import all the Waste Bin Images
imgBinList = []         #this list will contain all the 4 Bins
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins) #variable mentin
for path in pathList:
    imgBinList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

while True:
    _, img = cap.read()

    imgBackground = cv2.imread("Resources/background.png")
    imgResize = cv2.resize(img, (454, 340))

    prediction = Classifier.getPrediction(img)
    print(prediction)

    classID = prediction[1]

    if classID != 0:     #in this even if get new prediction old prediction will also be present
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127)) #static image to dynamic Image
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]


    imgBackground = cvzone.overlayPNG(imgBackground, imgBinList[classIDBin], (895, 374))



    # Height and width of the output image
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display the output
    #cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
