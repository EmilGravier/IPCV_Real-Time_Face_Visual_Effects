import cv2 as cv

stream = cv.VideoCapture(0)
cascadeClassifier = cv.CascadeClassifier(r'classifier_files\haarcascade_frontalface_default.xml')



if not stream.isOpened():
    print('Stream unavailable')
    exit()

while True:
    ret, frame = stream.read()
    if not ret:
        print('Stream unavailable')
        break
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    faces = cascadeClassifier.detectMultiScale(grayFrame, 1.3, 5)

    for (x,y,w,h) in faces:
         cv.rectangle(grayFrame,(x,y),(x+w,y+h),(255,0,0),2)






    cv.imshow('Webcam', frame)
    if cv.waitKey(1) == ord('q'):
        break

stream.release()
cv.destroyAllWindows()