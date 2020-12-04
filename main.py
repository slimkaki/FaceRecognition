import cv2, face_recognition, json
import numpy as np
from FaceDetect import FaceDetect
import sys

if __name__ == '__main__':
    
    face = FaceDetect()

    if len(sys.argv) == 2 and sys.argv[1] == 'register':
        print('\033[95m.:Realizar Cadastro:.\033[0m')
        face.register()

    print('\033[95m.:Face Recon:.\033[0m')
    face.run()
    


