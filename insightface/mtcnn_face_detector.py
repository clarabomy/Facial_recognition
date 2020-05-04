import arrow
import argparse

import numpy as np
import mxnet as mx
from eyewitness.config import BoundedBoxObject
from eyewitness.detection_utils import DetectionResult
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import swap_channel_rgb_bgr, Image
from eyewitness.image_id import ImageId
from bistiming import SimpleTimer

from deploy.mtcnn_detector import MtcnnDetector


class MtcnnFaceDetector(ObjectDetector):
    def __init__(self, mtcnn_path, ctx):
        self.face_detector = MtcnnDetector(
            model_folder=mtcnn_path,
            ctx=ctx, num_worker=1, accurate_landmark=True,
            threshold=[0.6, 0.7, 0.8])

    def detect(self, image_obj, label='face', train=True):
        detected_objects = []
        frame = swap_channel_rgb_bgr(np.array(image_obj.pil_image_obj))
        ret = self.face_detector.detect_face(frame, det_type=0)
        if ret:
            bbox, _ = ret

            # boundingboxes shape n, 5
            for idx in range(bbox.shape[0]):
                x1, y1, x2, y2, score = bbox[idx]
                if train: 
                    score=1
                detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, label, score, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    def detect_frame(self, frame):
        detected_objects = []
        # frame = swap_channel_rgb_bgr(np.array(image_obj.pil_image_obj))
        ret = self.face_detector.detect_face(frame, det_type=0)
        #print(ret)
        if ret:
            bbox, _ = ret

            # boundingboxes shape n, 5
            for idx in range(bbox.shape[0]):
                x1, y1, x2, y2, score = bbox[idx]
                detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, 'face', score, ''))
            
        image_dict = {
            'image_id': 'frame',
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return(detection_result)

    @property
    def valid_labels(self):
        return set(['face'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    args = parser.parse_args()

    if args.gpu >= 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu(0)
    model_name = 'MTCNN'
    with SimpleTimer("Loading model %s" % model_name):
        face_detector = MtcnnFaceDetector(args.mtcnn_path, ctx)

    raw_image_path = 'demo/183club/test_image.jpg'
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)

    with SimpleTimer("detect a img %s" % model_name):
        detection_results = face_detector.detect(train_image_obj)

    for obj in detection_results.detected_objects:
        print(obj)
