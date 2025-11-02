import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import os

stream = cv.VideoCapture(0)
cascadeClassifier = cv.CascadeClassifier(r'classifier_files\haarcascade_frontalface_default.xml')

if not stream.isOpened():
    print('Stream unavailable')
    exit()

#mediapipe hand detection
mediapipe_hands = mp.solutions.hands
#number of hands, full complexity model, minimum detection confidence threshold, minimum tracking confidence threshold
hands = mediapipe_hands.Hands(max_num_hands=1,model_complexity=1,min_detection_confidence=0.8,min_tracking_confidence=0.5)

#EMOJIS
emoji_folder = 'emojis'
#load emoji_0.png to emoji_4.png for 1 to 5 fingers
emojis = [cv.imread(os.path.join(emoji_folder, f'emoji_{i}.png'), cv.IMREAD_UNCHANGED) for i in range(5)]

#mediapipe finger landmarks
TIP_IDS = [4, 8, 12, 16, 20]  #tip landmarks of: thumb, index finger, middle finger, ring finger, pinky finger
JOINT_IDS = [2, 6, 10, 14, 18]  #middle joint landmarks (same finger order as above)

#drawing the emoji overlayed in frame at top of face detection boundary box (video frame, what emoji to use, emoji coordinates, emoji angle, size)
def overlay_emoji(frame, emoji_img, center, angle, size):
    #resize the emoji
    emojiResized = cv.resize(emoji_img, (size, size), interpolation=cv.INTER_AREA)
    #get height and width of resized emoji
    h, w = emojiResized.shape[:2]
    #rotation matrix to rotate the emoji by the computed angle later
    rotationMatrix = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    #2D affine transformation to actually "create" the rotated emoji, flags are used to somewhat avoid artifacts created by rotation
    rotated = cv.warpAffine(emojiResized, rotationMatrix, (w, h),flags=cv.INTER_AREA,borderMode=cv.BORDER_CONSTANT,borderValue=(0, 0, 0, 0))

    #if emoji has alpha channel
    if rotated.shape[2] == 4:  
        #get the transparency value
        alpha = rotated[:, :, 3] / 255.0
        #actually display the emoji information looping over each of the 3 color channels at correct location in each video frame
        for c in range(3):
            x, y = center
            #top left corner of output location
            x1, y1 = int(x - w//2), int(y - h//2)
            #bottom right corner of output location
            x2, y2 = x1 + w, y1 + h
            #check if emoji is outside of video frame, if not, skip displaying
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            #display the emoji at the correct location using the alpha channel for blending
            frame[y1:y2, x1:x2, c] = (alpha * rotated[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c])
    
    #same thing but for emojis without alpha channel           
    else:
        x, y = center
        x1, y1 = int(x - w//2), int(y - h//2)
        x2, y2 = x1 + w, y1 + h
        frame[y1:y2, x1:x2] = rotated

#main loop
while True:
    #read webcam stream
    ret, frame = stream.read()
    if not ret:
        print('Stream unavailable')
        break

    #face detection
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = cascadeClassifier.detectMultiScale(grayFrame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #convert current frame to rg for mediapipe
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #detect the hand landmarks for the current frame
    result = hands.process(frame_rgb)

    #initialize amount of fingers and starting angle
    fingers_up = 0
    angle_deg = 0

    #only proceed if there is at least 1 hand
    if result.multi_hand_landmarks:
        #get all the landmarks of the first hand
        hand_landmarks = result.multi_hand_landmarks[0]
        #get height and width of the frame
        h_img, w_img, _ = frame.shape
        #extract the list of landmarks
        landmarks = hand_landmarks.landmark
        #empty list for fingers that are actually extended
        extended_points = []

        #convert normalized landmark coordinates to pixel coordinates, looping over each finger
        for tip_id, joint_id in zip(TIP_IDS, JOINT_IDS):
            tip = landmarks[tip_id]
            joint = landmarks[joint_id]
            tip_x, tip_y = int(tip.x * w_img), int(tip.y * h_img)
            joint_x, joint_y = int(joint.x * w_img), int(joint.y * h_img)

            #for the thumb, compare position horizontally if thumb tip is extended further than thumb joint
            if tip_id == 4:
                if tip_x > joint_x:
                    #increase the count of total extended or "up" fingers
                    fingers_up += 1
                    extended_points.append((tip_x, tip_y))
            #compare vertically if fingertips are extended further than the respective finger joint
            else:
                if tip_y < joint_y:
                    #increase the count of total "up" fingers
                    fingers_up += 1
                    extended_points.append((tip_x, tip_y))

            #draw a green circle on the fingertip location
            cv.circle(frame, (tip_x, tip_y), 5, (0, 255, 0), -1)

        if extended_points:
            #compute centoid of extended fingers
            points_np = np.array(extended_points)
            cx, cy = np.mean(points_np, axis=0).astype(int)
            #draw a blue circle at the location of centroid
            cv.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

            #get wrist landmark
            wrist = landmarks[0]
            wx, wy = int(wrist.x * w_img), int(wrist.y * h_img)
            #draw a blue circle at the location of wrist landmark
            cv.circle(frame, (wx, wy), 8, (255, 0, 0), -1)

            #calculate vector from wrist to the finger centroid to get hand rotation angle
            dx, dy = cx - wx, cy - wy
            angle_rad = math.atan2(dx, -dy)#invert y-axis
            angle_deg = math.degrees(angle_rad)

            #draw yellow line from wrist to finger centroid
            cv.line(frame, (wx, wy), (cx, cy), (0, 255, 255), 3)

    #actually overlaying the emojis if 3 conditions are met:
    #1 hand needs to be detected, a face needs to be detected, at least 1 finger needs to be detected
    if result.multi_hand_landmarks and len(faces) > 0 and fingers_up > 0:
        #positioning relative to detected face boundary box
        for (x, y, w, h) in faces:
            #getting the correct emoji png
            emoji_index = fingers_up - 1 
            emoji_img = emojis[emoji_index]
            #calculate where to place the emoji (centered and slightly above face)
            center_x = x + w // 2
            center_y = y - 40
            #actually draw the emoji at the correct location
            overlay_emoji(frame, emoji_img, (center_x, center_y), angle_deg, size=80)

    #display hand angle in frame
    cv.putText(frame, f"Hand Angle: {angle_deg:.1f} deg", (10, 40),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #display amount of fingers that are up in frame
    cv.putText(frame, f"Fingers Up: {fingers_up}", (10, 80),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #show the webcame frame
    cv.imshow('Webcam', frame)
    if cv.waitKey(1) == ord('q'):
        break

#close everything
stream.release()
cv.destroyAllWindows()
hands.close()
