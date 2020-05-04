import numpy as np
import mxnet as mx
import os, sys
from PIL import Image as Img
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from mtcnn_face_detector import MtcnnFaceDetector
from arcface_objects_classifier import ArcFaceClassifier, train_arcface_model
from video_stream import video_stream
from eyewitness.config import BoundedBoxObject
from helpers.args import get_args
from helpers.utils import load_json


def main(config): 
    face_detector = MtcnnFaceDetector(mtcnn_path=config["mtcnn_model_path"], ctx=mx.gpu(0))

    if  config["train_recognition_model"]:
        print(f"[LOG] Train the recognition model")
        arcface_classifier = train_arcface_model(face_detector, config, 
                                    dataset_folder=config["data"]["dataset_path"],
                                    embedding_folder=config["data"]["embedding_path"]) 

    else:
        faces_embedding_file = config["data"]["embedding_path"]
        embedding, registered_ids = ArcFaceClassifier.restore_embedding_info(faces_embedding_file)
        arcface_classifier = ArcFaceClassifier(config, registered_ids, registered_images_embedding=embedding)
    
    video_stream(face_detector, arcface_classifier, unknown_folder=config["data"]["unknown_path"])


if __name__ == "__main__":
    args = get_args()
    
    if args.config is None or not os.path.exists(args.config):
        print(f"[ERROR] Unable to read the config file : {args.config}. Please specify a correct path for the config file.")
        sys.exit(0)
    
    config_file = load_json(args.config)
    
    main(config_file)
