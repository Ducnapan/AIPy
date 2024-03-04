import cv2

# define a video capture object
cap = cv2.VideoCapture(0)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

'''
    # if you want to detect any object for example eyes, use one more layer of classifier as below:
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
'''

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circle_center = (img.shape[1] // 2, img.shape[0] // 2)
    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if x < circle_center[0] < x + w and y < circle_center[1] < y + h:
            cv2.putText(img, 'Circle inside rectangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif circle_center[0] < x:
            cv2.putText(img, 'Circle on the left of rectangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif circle_center[0] > x + w:
            cv2.putText(img, 'Circle on the right of rectangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
       



    cv2.circle(img, circle_center, radius=10, color=(0, 0, 255), thickness=1)
    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyWindow('face_detect')




