import cv2, face_recognition, json
import numpy as np
from FaceDetect import FaceDetect

if __name__ == '__main__':

    # Obama_image = face_recognition.load_image_file("Obama.jpg")
    # Obama_face_encoding = face_recognition.face_encodings(Obama_image)[0]
    # # print(Obama_face_encoding)

    # file = open('faces.json')
    # conteudo = file.read()
    # db = json.loads(conteudo)
    # file.close()

    # print(db)
    # # print(np.array(db["Obama"]["encoding"]))

    # db["Obama"] = {}
    # db["Obama"]["encoding"] = list(Obama_face_encoding)
    # db["Obama"]["infos"] = {}

    # file = open('faces.json', 'w')
    # conteudo = json.dumps(db, indent=4, sort_keys=True)
    # file.write(conteudo)
    # file.close()

    face = FaceDetect()
    face.run()
    


