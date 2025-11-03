import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import os

stream = cv.VideoCapture(0)
cascadeClassifier = cv.CascadeClassifier(r'classifier_files\\haarcascade_frontalface_default.xml')

if not stream.isOpened():
    print('Stream unavailable')
    exit()


#mediapipe hand tracking solution, only used to detect the hand landmark keypoints
mediapipeHands = mp.solutions.hands
#setup for hands object with these parameters(max 1 hand at once, max complexity model, minimum hand detection confidence score, minimum tracking confidence score to keep tracking a keypoint)
hands = mediapipeHands.Hands(max_num_hands=1,model_complexity=1,min_detection_confidence=0.8,min_tracking_confidence=0.5)

#Load the emojis from folder
#folder name
emojiFolder = 'emojis'
#empty list to hold emojis
emojis = []
#loading the 5 different emojis for up to 5 fingers held up
for i in range(5):
    #constructing the filename
    filename = f"emoji_{i}.png"
    #constructing the filepath
    path = os.path.join(emojiFolder, filename)
    #load the current image at the filepath, cv.IMREAD_UNCHANGED to keep alpha channel
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    #add emoji image to the list
    emojis.append(img)

#fingertip landmarks from mediapipe
tipIDs = [4, 8, 12, 16, 20]
#finger base joint landmarks
baseIDs = [1, 5, 9, 13, 17]
#finger middle joints (2 each) landmarks
middleJointIDs = [2,3, 6,7, 10,11, 14,15, 18,19]
#wrist landmark
wristID = [0]

#function to overlay and draw emojis in the frame
def overlayEmoji(frame, emoji_img, center, angle, size):
    #resize the image, cv.INTER_AREA for interpolation (helps with shrinking)
    emojiResized = cv.resize(emoji_img, (size, size), interpolation=cv.INTER_AREA)
    #get the new height and width of the image
    h, w = emojiResized.shape[:2]
    #affine rotation matrix to rotate around the emojis center
    rotationMatrix = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    #apply the affine transform to rotate (again interpolation and borderMode=cv.BORDER_CONSTANT to make parts outside the image transparent black (0,0,0,0))
    rotated = cv.warpAffine(emojiResized, rotationMatrix, (w, h),flags=cv.INTER_AREA, borderMode=cv.BORDER_CONSTANT,borderValue=(0,0,0,0))
   
    # Check if the emoji has an alpha channel
    if rotated.shape[2] == 4:
        #extract alpha channel and normalize to [0,1] for blending with frame
        alphaChannel = rotated[:, :, 3] / 255.0

        #get desired x and ycenter coordinates of emoji
        centerX, centerY = center
        #compute x coordinate of top left corner on the frame
        topLeftX = int(centerX - w // 2)
        #compute y coordinate of top left corner on the frame
        topLeftY = int(centerY - h // 2)
        #compute x coordinate of bottom right corner
        bottomRightX = topLeftX + w
        #compute y coordinate of bottom right corner
        bottomRightY = topLeftY + h

        #if emoji is completely in frame boundaries
        if topLeftX >= 0 and topLeftY >= 0 and bottomRightX <= frame.shape[1] and bottomRightY <= frame.shape[0]:
            #loop over the 3 color channels
            for channel in range(3):
                #extract region of the frame where the emoji will be placed
                frameRegion = frame[topLeftY:bottomRightY, topLeftX:bottomRightX, channel]
                #get the current emoji color channel
                emojiChannel = rotated[:, :, channel]
                #blend the emoji with the frame region using alpha transparency
                blendedResult = (alphaChannel * emojiChannel) + ((1 - alphaChannel) * frameRegion)
                #place the blended result back into the frame
                frame[topLeftY:bottomRightY, topLeftX:bottomRightX, channel] = blendedResult

#calculating the palm center point from the mediapipe hand landmarks respective to the frame dimensions
def palmCenter(landmarks, frame_width, frame_height):
    #needed landmarks to find palm; wrist landmark and base of each finger
    palmIndices = [0, 1, 5, 9, 13, 17]
    
    #list to store pixel coordinates 
    palmPoints = []
    #loop over the landmarks
    for idx in palmIndices:
        lm = landmarks[idx]
        #convert to pixel coordinates
        xPixel = lm.x * frame_width
        yPixel = lm.y * frame_height
        #store pixel coordinates in list
        palmPoints.append((xPixel, yPixel))
    
    #convert the list to np array
    palmPoints_np = np.array(palmPoints)
    
    #palm center x and y is at average of the coordinates
    centerX, centerY = np.mean(palmPoints_np, axis=0)
    
    #return the x and y coordinates of the calculated center
    return int(centerX), int(centerY)

#function checking if a finger is extended or not
def fingerExtendedCheck(palmPoint, baseJointPoint, middleJointPoint, topJointPointPoint, fingertipPoint):
    
    # Compute distances from each finger joint to palm center
    distanceTip = np.linalg.norm(fingertipPoint - palmPoint)
    distanceTopJoint = np.linalg.norm(topJointPointPoint - palmPoint)
    distanceMiddleJoint = np.linalg.norm(middleJointPoint - palmPoint)
    distanceBaseJoint = np.linalg.norm(baseJointPoint - palmPoint)
    
    #Finger is extended if the distances decrease from tip to base
    isExtended = distanceTip > distanceTopJoint > distanceMiddleJoint > distanceBaseJoint
    
    return isExtended

#function checking if thumb is extended
def thumbExtendedCheck(palmPoint, thumbBaseJointPoint, thumbMiddleJointPoint, thumbTipPoint, wristPoint, imageHeight):
    #compute distances from thumb joints to palm center
    distanceThumbTip = np.linalg.norm(thumbTipPoint - palmPoint)
    distanceThumbMiddleJoint  = np.linalg.norm(thumbMiddleJointPoint  - palmPoint)
    distanceThumbBaseJoint = np.linalg.norm(thumbBaseJointPoint - palmPoint)

    #check if distances follow correct order: tip farthest, middle joint in middle, base closest
    isThumbOrdered = distanceThumbTip > distanceThumbMiddleJoint > distanceThumbBaseJoint

    #fist safety check: prevent detecting thumb as extended when it is closed in palm (silly way to do this, depends on distance from webcam)
    thumbNotInFist = distanceThumbTip > (imageHeight * 0.08)  #8% of image height

    # 4. Thumb is extended only if both conditions are satisfied
    return isThumbOrdered and thumbNotInFist

#main loop
while True:
    #capture frame from webcam, success is true if read correctly
    success, frame = stream.read()
    if not success:
        print("Webcam frame not available, exiting...")
        break

    #convert to grayframe for Haar cascade classifier
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #detect face using Haar cascade classifier
    detectedFaces = cascadeClassifier.detectMultiScale(grayFrame, scaleFactor=1.3, minNeighbors=5)

    #convert to BRG to RGB for mediapipe
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #run mediapipe hand detector and keypoint tracker on rgb frame
    handResults = hands.process(frameRGB)

    #counter for extended total fingers & variable to store hand angle
    fingersUp = 0
    handAngle = 0

    #if at least 1 hand is detected
    if handResults.multi_hand_landmarks:
        #get first detected hand
        handLandmarks = handResults.multi_hand_landmarks[0]
        #get frame dimensions
        imageHeight, imageWidth, _ = frame.shape
        #get the landmarks from the detected hand
        landmarks = handLandmarks.landmark

        #compute the palm center
        palmX, palmY = palmCenter(landmarks, imageWidth, imageHeight)
        #store the palm center as np array for distance calculations
        palmPoint = np.array([palmX, palmY])

        #convert wrist landmark from normalized coordinates to pixel coordinates.
        wristX, wristY = int(landmarks[0].x * imageWidth), int(landmarks[0].y * imageHeight)
        #store wrist as a np array for more calculations
        wristPoint = np.array([wristX, wristY])

        #checking each finger
        for fingerIndex in range(5):
            
            #convert to pixel coordinates for fingertip
            tipPoint = np.array([landmarks[tipIDs[fingerIndex]].x * imageWidth,landmarks[tipIDs[fingerIndex]].y * imageHeight])

            #thumb logic
            if fingerIndex == 0:
                #convert to pixel coordinates for thumb base point
                thumbBasePoint = np.array([landmarks[1].x * imageWidth, landmarks[1].y * imageHeight])
                #convert to pixel coordinates for thumb middle joint point
                thumbMiddleJointPoint  = np.array([landmarks[3].x * imageWidth, landmarks[3].y * imageHeight])
                #check if thumb is extended
                if thumbExtendedCheck(palmPoint, thumbBasePoint, thumbMiddleJointPoint, tipPoint, wristPoint, imageHeight):
                    fingersUp += 1
            #other fingers logic
            else:
                #convert to pixel coordinates for base joint
                fingerBaseJointPoint = np.array([landmarks[baseIDs[fingerIndex]].x * imageWidth,landmarks[baseIDs[fingerIndex]].y * imageHeight])
                #convert to pixel coordinates for middle joint
                fingerMiddleJointPoint = np.array([landmarks[baseIDs[fingerIndex] + 1].x * imageWidth,landmarks[baseIDs[fingerIndex] + 1].y * imageHeight])
                #convert to pixel coordinates for top joint
                fingerTopJointPoint = np.array([landmarks[baseIDs[fingerIndex] + 2].x * imageWidth,landmarks[baseIDs[fingerIndex] + 2].y * imageHeight])
                #check if finger is extended and increase counter if yes
                if fingerExtendedCheck(palmPoint, fingerBaseJointPoint, fingerMiddleJointPoint, fingerTopJointPoint, tipPoint):
                    fingersUp += 1

        #compute hand rotation 
        delta_x, delta_y = palmX - wristX, palmY - wristY
        handAngle = math.degrees(math.atan2(-delta_x, -delta_y))

        #landmark colors (BGR)
        colorWrist = (0, 0, 255) #red
        colorBase  = (0, 140, 255) #orange
        colorMiddle = (0, 255, 255) #yellow
        colorTip   = (0, 255, 0) #green
        colorPalm  = (0, 0, 0) #black

        #draw the landmarks on screen
        for i, landmark in enumerate(landmarks):
            #convert to pixel coordinates
            lx, ly = int(landmark.x * imageWidth), int(landmark.y * imageHeight)

            #draw colored circles at the respective landmarks (-1 for filled circle)
            if i in wristID:
                cv.circle(frame, (lx, ly), 7, colorWrist, -1)
            elif i in baseIDs:
                cv.circle(frame, (lx, ly), 6, colorBase, -1)
            elif i in middleJointIDs:
                cv.circle(frame, (lx, ly), 5, colorMiddle, -1)
            elif i in tipIDs:
                cv.circle(frame, (lx, ly), 8, colorTip, -1)

        #draw palm center
        cv.circle(frame, (palmX, palmY), 7, colorPalm, -1)
        #draw line from wrist to palm center
        cv.line(frame, (wristX, wristY), (palmX, palmY), (255, 255, 0), 3) #turquoise

    #show emoji if face detected and fingers counted
    if handResults.multi_hand_landmarks and len(detectedFaces) > 0 and fingersUp > 0:
        #bounding box from face detection
        for (x, y, w, h) in detectedFaces:
            #which emoji to choose, min to ensure we don't exceed emojis available
            emojiIndex = min(fingersUp - 1, 4)
            #get the emoji
            emojiImage = emojis[emojiIndex]
            #emoji coordinates slightly above center of face detection boundary box
            emojiCenter = (x + w // 2, y - 40)
            #draw the emoji
            overlayEmoji(frame, emojiImage, emojiCenter, handAngle, 80)

    #display the hand angle on screen
    cv.putText(frame, f"Hand Angle: {handAngle:.1f}", (10, 40),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #display the amount of raised fingers & thumbs on screen
    cv.putText(frame, f"Fingers Up: {fingersUp}", (10, 80),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv.imshow('Webcam', frame)
    if cv.waitKey(1) == ord('q'):
        break

stream.release()
cv.destroyAllWindows()
hands.close()
