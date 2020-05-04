import os, json


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
