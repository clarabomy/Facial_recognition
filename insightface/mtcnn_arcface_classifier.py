import argparse
import arrow

import mxnet as mx
from eyewitness.image_utils import ImageHandler, Image
from eyewitness.dataset_util import BboxDataSet
from eyewitness.object_detector import ObjectDetector
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from bistiming import SimpleTimer

from arcface_objects_classifier import ArcFaceClassifier, generate_dataset_arcface_embedding, train_arcface_model
from mtcnn_face_detector import MtcnnFaceDetector


class MtcnnArcFaceClassifier(ObjectDetector):
    def __init__(self, args, registered_ids_trans=None, similarity_threshold=0.40):
        
        ctx = mx.gpu(0)
        mtcnn_path = 'deploy/mtcnn-model/'
        model_name = 'MTCNN'
        with SimpleTimer("Loading model %s" % model_name):
            self.face_detector = MtcnnFaceDetector(mtcnn_path, ctx)

        embedding, registered_ids = ArcFaceClassifier.restore_embedding_info(
            args.dataset_embedding_path)
        if registered_ids_trans is not None:
            self.registered_ids = [registered_ids_trans[i] for i in registered_ids]
        else:
            self.registered_ids = registered_ids

        self.arcface_classifier = ArcFaceClassifier(
            self.registered_ids, registered_images_embedding=embedding,
            threshold=similarity_threshold)

    def detect(self, image_obj):
        with SimpleTimer("Detect face with mtcnn"):
            face_detection_result = self.face_detector.detect(image_obj)
        if face_detection_result.detected_objects:
            with SimpleTimer("classify faces with arcface"):
                detection_result = self.arcface_classifier.detect(
                    image_obj, face_detection_result.detected_objects)
        else:
            image_dict = {
                'image_id': 'face',
                'detected_objects': [],
            }
            detection_result = DetectionResult(image_dict)
        return detection_result

    @property
    def valid_labels(self):
        return set(i for i in self.registered_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model example')
    # model config
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='models/model-r100-ii/model,0',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                        help='path to load model.')
    # dataset, embedding setting
    parser.add_argument('--dataset_folder', default=None,
                        help='dataset used to registered')
    parser.add_argument('--dataset_embedding_path', default='faces.pkl',
                        help='pre-generated arcface embedding')
    # image used to test
    parser.add_argument('--demo_image', default='demo/p.jpg',
                        help='dataset used to registered')
    parser.add_argument('--drawn_image_path', default='detected_image/demo_ben.jpg',
                        help='the image output path with drawn the detection result')

    args = parser.parse_args()

    if args.dataset_folder is not None:
        # mtcnn_path = 'deploy/mtcnn-model/'
        # face_detector = MtcnnFaceDetector(mtcnn_path, ctx=mx.gpu(0))
        dataset_name = 'faces'
        faces_dataset = BboxDataSet(args.dataset_folder, dataset_name)
        generate_dataset_arcface_embedding(faces_dataset, args.dataset_embedding_path)
        #train_arcface_model(face_detector, args.dataset_folder, dataset_name)
    
    mtcnn_arcface_classifier = MtcnnArcFaceClassifier(args)
    raw_image_path = args.demo_image

    if raw_image_path:
        test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
        test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)
        detection_result = mtcnn_arcface_classifier.detect(test_image_obj)
        print(detection_result.detected_objects)
        if args.drawn_image_path:
            ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
            ImageHandler.save(test_image_obj.pil_image_obj, args.drawn_image_path)
