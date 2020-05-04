import argparse
import pickle
import logging
from pathlib import Path
import os
import arrow
import numpy as np
from eyewitness.config import BoundedBoxObject
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from eyewitness.object_classifier import ObjectClassifier
from eyewitness.detection_utils import DetectionResult
from eyewitness.config import DATASET_ALL
from bistiming import SimpleTimer
from dchunk import chunk_with_index
from insightface.deploy import face_model
from mtcnn_face_detector import MtcnnFaceDetector


class ArcFaceClassifier(ObjectClassifier):
    def __init__(self, registered_ids, 
                 objects_frame=None, registered_images_embedding=None,
                 threshold=0.55, batch_size=20):

        params =  {
            'model': 'insightface/models/model-r34-ii/model,0',
            'image_size': '112,112',
            'gpu': 0,
        }

        self.model = face_model.FaceModel(params)
        self.dataset_folder = "dataset"
        self.image_size = [int(i) for i in params['image_size'].split(',')]
    
        if registered_images_embedding is not None:
            self.registered_images_embedding = registered_images_embedding
        
        else:
            n_images = objects_frame.shape[0]
            registered_images_embedding_list = []
            for row_idx, batch_start, batch_end in chunk_with_index(range(n_images), batch_size):
                objects_embedding = self.model.get_faces_feature(
                    objects_frame[batch_start: batch_end])
                registered_images_embedding_list.append(objects_embedding)
            self.registered_images_embedding = np.concatenate(registered_images_embedding_list)

        self.registered_ids = registered_ids
        self.threshold = threshold
        self.unknown = 'Unknown'


    def detect(self, image_obj, bbox_objs=None, batch_size=2):
        if bbox_objs is None:
            x2, y2 = image_obj.pil_image_obj.size
            bbox_objs = [BoundedBoxObject(0, 0, x2, y2, '', 0, '')]

        n_bbox = len(bbox_objs)
        result_objects = []
        for row_idx, batch_start, batch_end in chunk_with_index(range(n_bbox), batch_size):
            batch_bbox_objs = bbox_objs[batch_start:batch_end]
            objs = image_obj.fetch_bbox_pil_objs(batch_bbox_objs)
            objects_frame = resize_and_stack_image_objs(self.image_size, objs)
            objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
            objects_embedding = self.model.get_faces_feature(objects_frame)
            similar_matrix = objects_embedding.dot(self.registered_images_embedding.T)
            detected_idx = similar_matrix.argmax(1)
            for idx, bbox in enumerate(batch_bbox_objs):
                x1, y1, x2, y2, _, _, _ = bbox
                label = self.registered_ids[detected_idx[idx]]
                score = similar_matrix[idx, detected_idx[idx]]
                if score < self.threshold:
                    label = self.unknown
                result_objects.append(BoundedBoxObject(x1, y1, x2, y2, label, score, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': result_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result


    def valid_labels(self):
        return set(self.registered_ids + [self.unknown])


    def store_embedding_info(self, pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump((self.registered_images_embedding, self.registered_ids), file=f)
    

    def recognize_person(self, img, face_detection_result):
        image_id = ImageId(channel='stream', timestamp=arrow.now().timestamp)
        image_obj = Image(image_id, pil_image_obj=img)

        detection_result = self.detect(image_obj, face_detection_result.detected_objects)  
        persons = [(elt.label, elt.x1, elt.y1, elt.x2, elt.y2, elt.score) for elt in detection_result.detected_objects]
        
        return(persons, detection_result.detected_objects)


    @staticmethod
    def restore_embedding_info(pkl_path):
        pkl_path_obj = Path(pkl_path)
        if not pkl_path_obj.exists():
            raise Exception('path %s not exist' % pkl_path_obj)

        if pkl_path_obj.is_dir():
            pkls = pkl_path_obj.glob('*.pkl')
            all_face_embedding_list = []
            all_face_ids = []
            for pkl in pkls:
                with open(str(pkl), 'rb') as f:
                    face_embedding, face_ids = pickle.load(f)
                    if face_embedding.shape[0] != len(face_ids):
                        LOG.warn('the pkl %s, without same shape embedding and face ids', pkl)
                        continue
                    all_face_ids.extend(face_ids)
                    all_face_embedding_list.append(face_embedding)

            all_face_embedding = np.concatenate(all_face_embedding_list)
            assert all_face_embedding.shape[0] == len(all_face_ids)

            return all_face_embedding, all_face_ids
        else:
            with open(str(pkl_path), 'rb') as f:
                return pickle.load(f)


def train_arcface_model(face_detector, dataset_folder="data/datasets", output_path="data/faces_embedding"):
    objs = []
    register_image_bbox_objs = []
    registered_ids = []

    for folder in os.listdir(dataset_folder):
        for face in os.listdir(f"{dataset_folder}/{folder}"):
            raw_image_path = f"{dataset_folder}/{folder}/{face}" 
            train_image_id = ImageId(channel='train_img', timestamp=arrow.now().timestamp, file_format=os.path.splitext(face)[1])
            train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)

            try:
                face_detection_result = face_detector.detect(train_image_obj, label=folder)
                if len(face_detection_result.detected_objects) == 1:
                    register_image_bbox_objs.append(face_detection_result.detected_objects[0])
                    registered_ids.append(face_detection_result.detected_objects[0].label)
                    objs += train_image_obj.fetch_bbox_pil_objs(register_image_bbox_objs)
            except:
                print(f"[ERROR] An error occured with the face detector model. Can't open the photo {dataset_folder}/{folder}/{face}")
            register_image_bbox_objs = []

    objects_frame = resize_and_stack_image_objs((112, 112), objs)
    print("Object_frame shape:", objects_frame.shape)

    objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
    with SimpleTimer("[INFO] Extracting embedding for our dataset"):
        arcface_classifier = ArcFaceClassifier(registered_ids, objects_frame=objects_frame)

    embedding_path = f"{output_path}/faces.pkl"
    print(f"[INFO] Store face embedding to {embedding_path}")
    arcface_classifier.store_embedding_info(embedding_path)
    
    ids_path = f"{output_path}/registered_ids.txt"
    print(f"[INFO] Store registered ids to {ids_path}")
    with open(ids_path, 'w') as filehandle:
        for registered_id in registered_ids:
            filehandle.write('%s\n' % registered_id)

    return arcface_classifier







