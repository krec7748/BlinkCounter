import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

#Set DIR
PRACTICE_MODE = False #Infinite loop

VIDEO_DIR = "Video_DU.mp4"
OUTOUT_DIR = "./output.mp4"

cap = cv2.VideoCapture(VIDEO_DIR)   #Webcam: VIDEO_DIR = 0
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter(OUTOUT_DIR, fourcc, fps, (int(w * 1.5), h))

#Detector
detector = FaceMeshDetector(maxFaces = 1)
    
#Dector id list (Reference: https://github.com/google-ar/sceneform-android-sdk/issues/563)
idList = [22, 23, 24, 26, 110,      #bottom
          157, 158, 159, 160, 161,  #top
          130, 243]                 #left edge, right edge         

#Initial value of the ratio between horizontal value and vertical of the eye
ratio_list = list()

#Initial value of blink count
blinkCount = 0

#Initial value of frame counter
counter = 0

#Set color
purple_color = (255, 0, 255)
green_color = (0, 200, 0)

#Set ratio plot 
plotY = LivePlot(int(w / 2), h, [20, 50], invert = True)

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        if PRACTICE_MODE == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #영상이 멈추지 않음. (다시 첫 프레임으로 되돌아감.)
        else:
            break
    
    success, img = cap.read()  
    img, faces = detector.findFaceMesh(img, draw = False)
    
    if faces:
        face = faces[0] #maxFaces = 1
        for id in idList:
            cv2.circle(img, face[id], 4, purple_color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        
        #cv2.line(img, leftUp, leftDown, green_color, 3)
        #cv2.line(img, leftLeft, leftRight, green_color, 3)
        
        #Calculate ratio (lengthVer, lengthHor)
        ratio = int((lengthVer / lengthHor) * 100)
        ratio_list.append(ratio)
        if len(ratio_list) > 3:
            ratio_list.pop(0)
            
        ratioAvg = sum(ratio_list) / len(ratio_list)
        
        if ratioAvg < 35 and counter == 0:
            blinkCount += 1
            color = green_color
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 15:    #Wait frames after raising blinkCount 
                counter = 0
                color = purple_color
            
        cvzone.putTextRect(img, f"Blink Count: {blinkCount}", (50, 100), colorR = color)
        imgPlot = plotY.update(ratioAvg, color)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        imgStack = cvzone.stackImages([img, img], 2, 1)
    
    cv2.imshow("Image", imgStack)
    cv2.waitKey(25)
    
    out.write(imgStack)
    
out.release()
cv2.destroyAllWindows()
