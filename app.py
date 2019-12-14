import base64
from flask import Flask, request, jsonify
import numpy as np
import cv2

from face.app import Face
from face.exceptions import InvalidException


# Initialize the app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
face = Face()


# route http posts to this method
@app.route("/api/face", methods=['POST'])
def detect_face():
    try:
        req = request.json

        if req.get('image') is None:
            raise InvalidException('image is required.')

        # decode base64 string into np array
        nparr = np.frombuffer(base64.b64decode(req['image'].encode('utf-8')), np.uint8)

        # decoded image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise InvalidException('Unable to parse the image.')

        num, data = face.detect(img)
        response = {
            'success': True,
            'status code': 201,
            'message': '{} faces detected'.format(num),
            'data': {'image': data},
            }
        resp = jsonify(response)
        resp.status_code = 200
        return resp

    except Exception as e:
        response = {
            'success': False,
            'status code': 500,
            'message': str(e),
            }
        resp = jsonify(response)
        resp.status_code = 500
        return resp

if __name__ == '__main__':
    app.run()
