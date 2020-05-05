from helpers.utils import create_dir, get_distance, append_list_as_row
from mtcnn_face_detector import MtcnnFaceDetector
from arcface_objects_classifier import ArcFaceClassifier
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from imutils.video import FPS
from PIL import Image as Img
import numpy as np
import cv2, keyboard, uuid, os, datetime

MARGIN = 25

def valid_photo(x1, y1, x2, y2, w, h):
    return (x1 - MARGIN > 0 and y1 - MARGIN > 0 and x2 + MARGIN < w and y2 + MARGIN < h)

def video_stream(face_detector, arcface_classifier, unknown_folder, logs_folder):
    cap = cv2.VideoCapture(0)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    # Create data folders 
    create_dir(unknown_folder)
    create_dir(logs_folder)

    # Create logs file 
    log_file = logs_folder + "/" + datetime.datetime.now().strftime("%d-%m-%Y") + ".csv"

    fps = FPS().start()
    nb_frames = 0
    last_bbs = []


    while(True):
        ret, frame = cap.read()
        face_detection_result = face_detector.detect_frame(frame)
        pil_im = Img.fromarray(frame)

        if nb_frames % 3 == 0:
            persons, detected_persons = arcface_classifier.recognize_person(pil_im, face_detection_result)
            frame = np.array(pil_im)

        for i, person in zip(range(len(persons)), persons):
            
            # Si début du programme ou personne supplémentaire ou en moins, réinitialisation 
            if not last_bbs or len(last_bbs) != len(persons):
                last_bbs = [(0,0,0,0) for x in range(len(persons))]
            
            label, x1, y1, x2, y2, confidence = str(person[0]), int(person[1]), int(person[2]), int(person[3]), int(person[4]), float(person[5])
            
            data_to_save = [datetime.datetime.now().strftime("%d-%m-%Y"), 
                            datetime.datetime.now().strftime("%H:%M:%S"),
                            label,
                            "{:.2f}".format(confidence)]

            if label == 'Unknown': 
                if get_distance(last_bbs[i], (x1,y1,x2,y2)) != -1 and valid_photo(x1, y1, x2, y2, width, height):
                    
                    try:
                        id_unknown = uuid.uuid4()
                        to_crop = np.array(pil_im)
                        cropped = to_crop[y1-MARGIN:y2+MARGIN, x1-MARGIN:x2+MARGIN] #0.5*... à tester !
                        cv2.imwrite(f"{unknown_folder}/{id_unknown}.jpg", cropped)
                        data_to_save.append(id_unknown)
                        append_list_as_row(log_file, data_to_save)

                    except:
                        pass

                color = (0,0,255) #red color
            
            else:
                if get_distance(last_bbs[i], (x1,y1,x2,y2)) != -1 and valid_photo(x1, y1, x2, y2, width, height):
                    append_list_as_row(log_file, data_to_save)
                
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
