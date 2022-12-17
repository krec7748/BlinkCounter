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

#Initial value of frame frame counter
frameCounter = 0

#Set color
purple_color = (255, 0, 255)
green_color = (0, 200, 0)
color = None #Set initial_color

#Set initial live plot
plotY = LivePlot(int(w / 2), h, [20, 50], invert = True)

while True:
    
    #PRACTICE_MODE = True >> Initial loop
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        if PRACTICE_MODE == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            break
    
    #get frame img, face (result of face detector)
    success, img = cap.read()  
    img, faces = detector.findFaceMesh(img, draw = False)
    
    if faces:
        face = faces[0] #maxFaces = 1
        
        #Draw points of left eye
        for id in idList:
            if color is None:
                cv2.circle(img, face[id], 4, purple_color, cv2.FILLED)
            else:
                cv2.circle(img, face[id], 4, color, cv2.FILLED)
                
        #Extract major point of left eye
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        
        #Draw horizontal line, vertical line of left eye
        #cv2.line(img, leftUp, leftDown, green_color, 3)
        #cv2.line(img, leftLeft, leftRight, green_color, 3)
        
        #Calculate ratio & Get ratioAvg (lengthVer, lengthHor)
        ratio = int((lengthVer / lengthHor) * 100)
        ratio_list.append(ratio)
        if len(ratio_list) > 3:
            ratio_list.pop(0)
            
        ratioAvg = sum(ratio_list) / len(ratio_list)
        
        #If ratioAvg < 35: Count the blink by 1
        if ratioAvg < 35 and frameCounter == 0:  #
            blinkCount += 1
            color = green_color
            frameCounter = 1
        
        #Wait 15 frames after raising blinkCount (Do not counting of blink)
        if frameCounter != 0:
            frameCounter += 1
            if frameCounter > 15:    
                frameCounter = 0
                color = purple_color
        
        #Set Texter
        cvzone.putTextRect(img, f"Blink Count: {blinkCount}", (50, 100), colorR = color)
        
        #Draw live plot, Stack result (img, liveplot)
        imgPlot = plotY.update(ratioAvg, color)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else: #error
        imgStack = cvzone.stackImages([img, img], 2, 1)
    
    #Draw result (Stack result)
    cv2.imshow("Image", imgStack)
    
    #Set frame speed
    cv2.waitKey(25)
    
    #Update output
    out.write(imgStack)
    
#Save result & Finish
out.release()
cv2.destroyAllWindows()
