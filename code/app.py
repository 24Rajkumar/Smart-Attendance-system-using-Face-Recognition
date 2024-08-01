#we have to create new environment
#create a new environment by typing 
#conda create --name mlenv python==3.9' in anaconda prompt
#4) 'mlenv' is name of your environment, you can give anyother name if you want
#) after creating new environment type 'conda activate mlenv'
#to activate your environment
# USAGE

# import the necessary packages
# Used for Face Recognition

import face_recognition
# load the video and processthe video frame
import cv2
# import os.path
# read and write contents to CSV FIle
import os
# to get the time stamp
from datetime import datetime
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="templates")


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')


@app.route('/home', methods=['GET'])
def about():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def predict():
    # Load images.
    print("[INFO] quantifying faces...")
    image_1 = face_recognition.load_image_file("dataset/dhoni/Dh (9).jpg")
    image_1_face_encoding = face_recognition.face_encodings(image_1)[0]

    image_2 = face_recognition.load_image_file("dataset/modi/modi (10).jpg")
    image_2_face_encoding = face_recognition.face_encodings(image_2)[0]

    image_3 = face_recognition.load_image_file("dataset/yuvraj/yuvi (10).jpg")
    image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

    image_4 = face_recognition.load_image_file("dataset/sharukh/sha (2).jpg")
    image_4_face_encoding = face_recognition.face_encodings(image_4)[0]
    
    image_5 = face_recognition.load_image_file("dataset/Raj/Rj (9).jpg")
    image_5_face_encoding = face_recognition.face_encodings(image_5)[0]
    
    image_6 = face_recognition.load_image_file("dataset/Achala/achala3.jpg")
    image_6_face_encoding = face_recognition.face_encodings(image_6)[0]

    image_7 = face_recognition.load_image_file("dataset/Bhavya/Bhavya (3).jpg")
    image_7_face_encoding = face_recognition.face_encodings(image_7)[0]
    
    

    # Create arrays of known face encodings and their names
    known_face_encodings = [

        image_1_face_encoding,
        image_2_face_encoding,
        image_3_face_encoding,
        image_4_face_encoding,
        image_5_face_encoding,
        image_6_face_encoding,
        image_7_face_encoding

    ]
    known_face_names = [

        "dhoni",
        "modi",
        "sharukh",
        "yuvraj",
        "Raj",
        "Achala",
        "Bhavya",
    ]

    # Initializing  variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # Get a reference to webcam #0 (the default one)

    print("[INFO] starting video stream...")
    video_capture = cv2.VideoCapture(0)

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to
        # RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        name = ""
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                # Enter Tracking log
                logFile = open('log.txt', mode="a")
                # set the file pointer to end of the file
                pos = logFile.seek(0, os.SEEK_END)
                # If this is a empty log file then write the column headings
                if pos == 0:
                    logFile.write("Year,Month,Day,Time,Name,Attendance")
                # Set Date and Time
                ts = datetime.now()
                newDate = ts.strftime("%m-%d-%y")
                year = ts.strftime("%Y")
                month = ts.strftime("%m")
                day = ts.strftime("%d")

                time1 = ts.strftime("%H:%M:%S")
                info = "{},{},{},{},{},{}\n".format(year, month, day, time1, name, "Present")
                logFile.write(info)
                logFile.close()

        face_names.append(name)

        # process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX
            # Draw a box around the face
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return render_template("upload.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)