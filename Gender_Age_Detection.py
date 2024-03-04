import cv2


# Input image

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Importing Models and set mean values
face1 = "Age-Gender-Detection/opencv_face_detector.pbtxt"
face2 = "Age-Gender-Detection/opencv_face_detector_uint8.pb"
age1 = "Age-Gender-Detection/age_deploy.prototxt"
age2 = "Age-Gender-Detection/age_net.caffemodel"
gen1 = "Age-Gender-Detection/gender_deploy.prototxt"
gen2 = "Age-Gender-Detection/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Using models
# Face
face_net = cv2.dnn.readNet(face2, face1)

# age
age_net = cv2.dnn.readNet(age2, age1)

# gender
gen_net = cv2.dnn.readNet(gen2, gen1)

# Categories of distribution
la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
lg = ['Male', 'Female']
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor

    # drawing bounding box around face
    for i, (x, y, w, h) in enumerate(faces):
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Extracting face as per the faceBox
        face = img[max(0, y - 15):min(y + h + 15, img.shape[0] - 1),
               max(0, x - 15):min(x + w + 15, img.shape[1] - 1)]

        # Extracting the main blob part
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Prediction of gender
        gen_net.setInput(blob)
        genderPreds = gen_net.forward()
        gender = lg[genderPreds[0].argmax()]

        # Prediction of age
        age_net.setInput(blob)
        agePreds = age_net.forward()
        age = la[agePreds[0].argmax()]

        # Putting text of age and gender
        # At the top of box
        cv2.putText(img,
                    'Gender: {}'.format(gender),
                    (x - 150, y + 10),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (217, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img,
                    'Age: {}'.format(age),
                    (x - 150, y + 50),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (217, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Final results (otherwise)
# Loop for all the faces detected






