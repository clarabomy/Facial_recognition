import argparse

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    
    # argparser.add_argument(
    #     '-t', '--train',
    #     dest='do_train',
    #     type=bool,
    #     default=False,
    #     help='Train the recognition model')
    
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        default=None,
        type=str,
        help='The path to the configuration file')

    args = argparser.parse_args()

    return args