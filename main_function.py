import numpy as np
import mxnet as mx
from PIL import Image as Img
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from mtcnn_face_detector import MtcnnFaceDetector
from arcface_objects_classifier import ArcFaceClassifier, train_arcface_model
from video_stream import video_stream
from eyewitness.config import BoundedBoxObject



def main(train=False): 
    # Face detector model
    mtcnn_path = 'insightface/deploy/mtcnn-model/'
    face_detector = MtcnnFaceDetector(mtcnn_path, ctx=mx.gpu(0))

    if not train:
        faces_embedding_file = "data/faces_embedding/faces.pkl"
        embedding, registered_ids = ArcFaceClassifier.restore_embedding_info(faces_embedding_file)
        arcface_classifier = ArcFaceClassifier(registered_ids, registered_images_embedding=embedding)

    else:
        arcface_classifier = train_arcface_model(face_detector)    
    
    video_stream(face_detector, arcface_classifier)
    # persons = arcface_classifier.recognize_person("lolilol.jpg", face_detector, "detected_image/183club/test_moches.jpg")
    # print(persons)



if __name__ == "__main__":
    main(train=False)
