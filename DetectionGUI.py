import tkinter as tk
from PIL import ImageTk, Image
import cv2

# Start window
window = tk.Tk()

# Window settings
window.title("Detection App")
window.geometry("800x700")




# Camera
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
camera_frame = tk.Frame(master=window,height=480, width=640, bg="black")
camera_frame.pack()
camera_main = tk.Label(camera_frame)
camera_main.pack()
def video_stream(capturing):

    if capturing:
        _, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        faces = faceCascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        imgtk = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=imgtk)
        camera_main.imgtk = imgtk
        camera_main.configure(image=imgtk)
        camera_main.after(1, video_stream,capturing)






# Container
cont = tk.Frame(window,bg="#E8e7e8")
cont.pack(side=tk.BOTTOM,fill=tk.X)

# Logo
label = tk.Label(cont,text="RoboVision",foreground="blue", bg="#E8e7e8",padx=10, font=("ROG Fonts",20))
label.pack(side=tk.LEFT)

# Divider
col = tk.Frame(cont,width=1,height=100,bg="black")
col.pack(side=tk.LEFT)

# Buttons

record_button = tk.Button(cont,text="Stop", width=20,height=3,foreground="white",bg="#2596be",borderwidth=0,command=lambda:video_stream(capturing=False))
record_button.pack(side=tk.RIGHT,padx=10)

# Buttons
record_button1 = tk.Button(cont,text="Start", width=20,height=3,foreground="white",bg="#2596be",borderwidth=0,command=lambda:video_stream(capturing=True))
record_button1.pack(side=tk.RIGHT,padx=10)


# Main window loop

window.mainloop()
