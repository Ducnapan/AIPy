from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


def generate_frames():
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            circle_center = (img.shape[1] // 2, img.shape[0] // 2)
            faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)

            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                if x < circle_center[0] < x + w and y < circle_center[1] < y + h:
                    cv2.putText(img, 'Circle inside rectangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                2)
                elif circle_center[0] < x:
                    cv2.putText(img, 'Circle on the left of rectangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)
                elif circle_center[0] > x + w:
                    cv2.putText(img, 'Circle on the right of rectangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)

            cv2.circle(img, circle_center, radius=10, color=(0, 0, 255), thickness=1)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
