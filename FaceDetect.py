import cv2, face_recognition, json
import numpy as np

class FaceDetect(object):

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.db = {}
        self.load()

    def load(self):
        # Open Database
        file = open('faces.json')
        conteudo = file.read()
        self.db = json.loads(conteudo)
        file.close()

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [ self.db[x]["encoding"] for x in self.db ]

        self.known_face_names = [ self.db[x]["infos"]["Name"] for x in self.db ]
        

    def save(self):
        file = open('faces.json', 'w')
        conteudo = json.dumps(self.db, indent=4, sort_keys=True)
        file.write(conteudo)
        file.close()

    def register(self):
        # name = input("What is your name? ")
        # Tirar uma foto e salvar em /images
            # image = face_recognition.load_image_file(f"images/{name}.jpg")
            # face_encoding = face_recognition.face_encodings(image)[0]
        # salvar no db (x CONVERTER ENCODING PARA LIST x)
        pass
            
    def run(self):

        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(0)

        # Load DataBase
        self.load()

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()