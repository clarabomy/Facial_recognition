import cv2
import os
import arrow
import keyboard
import numpy as np
import mxnet as mx
from PIL import Image as Img
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from mtcnn_face_detector import MtcnnFaceDetector
from arcface_objects_classifier import ArcFaceClassifier, train_arcface_model, video_stream
from bistiming import SimpleTimer
from eyewitness.config import BoundedBoxObject



def main(): 
    # Face detector model
    mtcnn_path = 'deploy/mtcnn-model/'
    face_detector = MtcnnFaceDetector(mtcnn_path, ctx=mx.gpu(0))
    arcface_classifier = train_arcface_model(face_detector)     # Face recognizer model
    video_stream(face_detector, arcface_classifier)
    #persons = arcface_classifier.recognize_person("lolilol.jpg", face_detector, "detected_image/183club/test_moches.jpg")
    #print(persons)





    # raw_image_path = 'test.jpg'
    # test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    # test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)

    # face_detection_result = face_detector.detect(test_image_obj)
    # print(arcface_classifier.registered_ids[:])
    # with SimpleTimer("Predicting image with classifier"):
    #     detection_result = arcface_classifier.detect(
    #         test_image_obj, face_detection_result.detected_objects)
    
    # # Dessine les boxs et sauvegarde les images
    # ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
    # ImageHandler.save(test_image_obj.pil_image_obj, "detected_image/183club/test_moche_1.jpg")
    # return()


    #cap = cv2.VideoCapture(0)
    #create_dir("photos")

    #while(True):
        # Capture frame-by-frame
        #ret, frame = cap.read()
        #face_detection_result = face_detector.detect_frame(frame)
        #pil_im = Img.fromarray(frame)
        #ImageHandler.draw_bbox(pil_im, face_detection_result.detected_objects)
        
        # Display the resulting frame
        #cv2.imshow('frame', np.array(pil_im))

        #if keyboard.is_pressed('p'):
            #count=0
            #raw_image_path = "demo/photo%d.jpg" %count
            #cv2.imwrite(raw_image_path, frame)        
            #count += 1
            #frame_id = ImageId(channel='recognition', timestamp=arrow.now().timestamp, file_format='jpg')
            #frame_obj = Image(frame_id, raw_image_path=raw_image_path)
            #detection_result = arcface_classifier.detect(
            #frame_obj, face_detection_result.detected_objects)
            #print(detection_result.image_id)
            #print(detection_result.detected_objects)

            #ImageHandler.draw_bbox(frame_obj.pil_image_obj, detection_result.detected_objects)
            #ImageHandler.save(frame_obj.pil_image_obj, "detected_image/183club/test.jpg")

    #         # Test de reconnaissance
    # raw_image_path = 'demo/183club/test_imag2.jpg'
    # test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    # test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)

    # face_detection_result = face_detector.detect(test_image_obj)
    # with SimpleTimer("Predicting image with classifier"):
    #     detection_result = arcface_classifier.detect(
    #     test_image_obj, face_detection_result.detected_objects)
    
    # Dessine les boxs et sauvegarde les images
    # ImageHandler.draw_bbox(train_image_obj.pil_image_obj, register_image_bbox_objs)
    # ImageHandler.save(train_image_obj.pil_image_obj, "detected_image/183club/drawn_image_1.jpg")

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    # When everything done, release the capture
    #cap.release()
    #cv2.destroyAllWindows()
    #return(0)



    # test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    # test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)
    # face_detection_result = face_detector.detect(test_image_obj)

if __name__ == "__main__":
    main()
