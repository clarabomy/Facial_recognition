import os, json
from math import sqrt
from csv import writer


def create_dir(dir: str):
    """
    Create the directory specified by dir_path if not exists.
    Args:
        dir_path: the path to the directory
    Requires:
        dir_path is not None
    Returns:
        a boolean stating if the creation succeded
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            return(0)
    except Exception as err:
        print("Creating directories error: {0}".format(err))


def load_json(path: str):
    """
    Load a json formated object from file_path.
    Args:
        file_path: the path to the file
    Requires:
        filepath is not None and file_path exists
    Returns:
        The loaded object
    """
    with open(path) as json_file:
        o_file = json_file.read()
    return json.loads(o_file)


def get_file_content(path: str):
    registered_ids = []
    
    with open(path, 'r') as filehandle:
        for line in filehandle:
            registered_id = line[:-1]
            registered_ids.append(registered_id)
    
    return registered_ids


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    
    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_distance(box_a, box_b):

    if bb_intersection_over_union(box_a, box_b) == 0:
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b
        
        return sqrt((x1_b-x2_a)**2 +(y1_b-y2_a)**2)
    else:
        return(-1)

 
def append_list_as_row(file_name: str, list_of_elem: list):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

