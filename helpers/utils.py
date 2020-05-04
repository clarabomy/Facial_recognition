import os

def create_dir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            return(0)
    except Exception as err:
        print("Creating directories error: {0}".format(err))


def get_file_content(path):
    registered_ids = []
    
    with open(path, 'r') as filehandle:
        for line in filehandle:
            registered_id = line[:-1]
            registered_ids.append(registered_id)
    
    return registered_ids