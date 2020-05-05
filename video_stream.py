from helpers.utils import create_dir, get_distance
from mtcnn_face_detector import MtcnnFaceDetector
from arcface_objects_classifier import ArcFaceClassifier
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from imutils.video import FPS
from PIL import Image as Img
import numpy as np
import cv2, keyboard, uuid

MARGIN = 25

def valid_photo(x1, y1, x2, y2, w, h):
    return (x1 - MARGIN > 0 and y1 - MARGIN > 0 and x2 + MARGIN < w and y2 + MARGIN < h)

def video_stream(face_detector, arcface_classifier, unknown_folder):
    cap = cv2.VideoCapture(0)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    create_dir(unknown_folder)
    fps = FPS().start()
    nb_frames = 0
    last_bbs = []

    while(True):
        ret, frame = cap.read()
        face_detection_result = face_detector.detect_frame(frame)
        pil_im = Img.fromarray(frame)

        if nb_frames % 5 == 0:
            persons, detected_persons = arcface_classifier.recognize_person(pil_im, face_detection_result)
            frame = np.array(pil_im)

        for i, person in zip(range(len(persons)), persons):
            
            # Si début du programme ou personne supplémentaire ou en moins, réinitialisation 
            if not last_bbs or len(last_bbs) != len(persons):
                last_bbs = [(0,0,0,0) for x in range(len(persons))]
            
            label, x1, y1, x2, y2, confidence = str(person[0]), int(person[1]), int(person[2]), int(person[3]), int(person[4]), float(person[5])

            if label == 'Unknown':
                if get_distance(last_bbs[i], (x1,y1,x2,y2)) != -1 and valid_photo(x1, y1, x2, y2, width, height):
                    to_crop = np.array(pil_im)
                    cropped = to_crop[y1-MARGIN:y2+MARGIN, x1-MARGIN:x2+MARGIN] #0.5*... à tester !
                    try:
                        cv2.imwrite(f"{unknown_folder}/{uuid.uuid4()}.jpg", cropped)
                    except:
                        pass
                color = (0,0,255) #red color
            
            else:
                color = (0,255,0) #green color

            cv2.putText(frame, 
                f"{label} - {confidence:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if get_distance(last_bbs[i], (x1,y1,x2,y2)) != -1:
                last_bbs[i] = (x1, y1, x2, y2)

        cv2.imshow('frame', frame)

        nb_frames += 1
        if nb_frames == 101:
            nb_frames = 0

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fps.stop()
    cap.release()
    cv2.destroyAllWindows()
    
    print("[INFO] Elasped time {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS {:.2f}".format(fps.fps()))

    return(0)
