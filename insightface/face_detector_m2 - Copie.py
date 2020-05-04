import arrow
import argparse

import mxnet as mx
import numpy as np
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from eyewitness.config import BoundedBoxObject
from bistiming import SimpleTimer

from arcface_objects_classifier import ArcFaceClassifier
from mtcnn_face_detector import MtcnnFaceDetector

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='models/model-r100-ii/model,0', help='')
parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                    help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

if __name__ == '__main__':
    args = parser.parse_args()

    # if args.gpu >= 0:
    #     ctx = mx.gpu(args.gpu)
    # else:
    #     ctx = mx.cpu(0)
    model_name = 'MTCNN'
    with SimpleTimer("Loading model %s" % model_name):
        face_detector = MtcnnFaceDetector(args.mtcnn_path, ctx=mx.gpu(0))

    # Préprocess de l'image
    raw_image_path = 'datasets/amine/photo0.jpg'
    #ImageId is used to standardize the image_id format
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)
    register_image_bbox_objs = []
    registered_ids = []
    objs = []

    face_detection_result = face_detector.detect(train_image_obj, label="Amine")
    print(face_detection_result.detected_objects)
    for bbox in face_detection_result.detected_objects :
        register_image_bbox_objs.append(bbox)
        registered_ids.append(bbox.label)
    objs = train_image_obj.fetch_bbox_pil_objs(register_image_bbox_objs)
    print(objs)

    register_image_bbox_objs = []
    raw_image_path = 'datasets/madonna/httpsuploadwikimediaorgwikipediacommonsthumbaaMadonnaatthepremiereofIAmBecauseWeArejpgpxMadonnaatthepremiereofIAmBecauseWeArejpg.jpg'
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)
    face_detection_result = face_detector.detect(train_image_obj, label="Madonna")
    print(face_detection_result.detected_objects)
    for bbox in face_detection_result.detected_objects :
        register_image_bbox_objs.append(bbox)
        registered_ids.append(bbox.label)
    objs += train_image_obj.fetch_bbox_pil_objs(register_image_bbox_objs)
    print(objs)

    register_image_bbox_objs = []
    raw_image_path = 'datasets/madonna/httpssmediacacheakpinimgcomxdcfdcfedfaedadjpg.jpg'
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)
    face_detection_result = face_detector.detect(train_image_obj, label="Madonna")
    print(face_detection_result.detected_objects)
    for bbox in face_detection_result.detected_objects :
        register_image_bbox_objs.append(bbox)
        registered_ids.append(bbox.label)
    objs += train_image_obj.fetch_bbox_pil_objs(register_image_bbox_objs)
    print(objs)

    objects_frame = resize_and_stack_image_objs((112, 112), objs)
    objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
    print(registered_ids)
    # Création du classifier et réentrainement
    model_name = 'Arcface'
    with SimpleTimer("Loading model %s" % model_name):
        arcface_classifier = ArcFaceClassifier(registered_ids, objects_frame=objects_frame)

    # Test de reconnaissance
    raw_image_path = 'datasets/madonna/httpresizeparismatchladmediafrrffffffcentermiddleimgvarnewsstorageimagesparismatchpeopleazmadonnafreFRMadonnajpg.jpg'
    test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)

    face_detection_result = face_detector.detect(test_image_obj)
    with SimpleTimer("Predicting image with classifier"):
        detection_result = arcface_classifier.detect(
            test_image_obj, face_detection_result.detected_objects)
    
    # Dessine les boxs et sauvegarde les images
    ImageHandler.draw_bbox(train_image_obj.pil_image_obj, register_image_bbox_objs)
    ImageHandler.save(train_image_obj.pil_image_obj, "detected_image/183club/test_clara.jpg")

    ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(test_image_obj.pil_image_obj, "detected_image/183club/test_clara_1.jpg")



    # Test de reconnaissance
    raw_image_path = 'datasets/amine/IMG_20200114_164420.jpg'
    test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)

    face_detection_result = face_detector.detect(test_image_obj)
    with SimpleTimer("Predicting image with classifier"):
        detection_result = arcface_classifier.detect(
            test_image_obj, face_detection_result.detected_objects)
    
    # Dessine les boxs et sauvegarde les images
    ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(test_image_obj.pil_image_obj, "detected_image/183club/test_amine_0.jpg")


#face detection puis quand on appuie sur une touche, on enregistre le visage reconnu