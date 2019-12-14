import os
import base64
import cv2
import dlib

class Face():

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        face = self.detect_face(image)
        image = self.draw_cords_on_image(image, face)
        image = cv2.imencode('.jpg', image)[1]
        return len(face), base64.b64encode(image).decode('utf-8', 'strict')

    def detect_face(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = self.get_dlib_face_coords(rgb_img)
        return faces

    def get_dlib_face_coords(self, img):
        rects = self.detector(img, 1)
        result = []
        for r in rects:
            result.append(
                {
                    'x': r.left(),
                    'y': r.top(),
                    'w': r.right() - r.left(),
                    'h':r.bottom() - r.top()
                }
            )

        return result

    def draw_cords_on_image(self, image, face):
        for i in face:
            cv2.rectangle(image, (i['x'], i['y']), (i['x']+i['w'], i['y']+i['h']), (0, 0, 255), 2)
        return image
