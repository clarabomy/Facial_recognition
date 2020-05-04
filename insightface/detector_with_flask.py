import argparse
import os
import logging

from eyewitness.flask_server import BboxObjectDetectionFlaskWrapper
from eyewitness.config import BBOX
from eyewitness.detection_result_filter import FeedbackBboxDeNoiseFilter
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from peewee import SqliteDatabase

from mtcnn_arcface_classifier import MtcnnArcFaceClassifier
from line_detection_result_handler import LineAnnotationSender



parser = argparse.ArgumentParser(description='face model example')
# model config
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='models/model-r100-ii/model,0',
                    help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                    help='path to load model.')
# embedding setting
parser.add_argument('--dataset_embedding_path', default='face.pkl',
                    help='pre-generated arcface embedding')

# eyewitness flask wrapper setting
parser.add_argument('--db_path', type=str, default='::memory::',
                    help='the path used to store detection result records')
parser.add_argument('--detector_host', type=str,
                    default='localhost', help='the ip address of detector')
parser.add_argument('--detector_port', type=int, default=5566, help='the port of detector port')
parser.add_argument('--drawn_image_dir', type=str, default=None,
                    help='the path used to store drawn images')


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def raw_image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    raw_image_path = drawn_image_path.replace('detected_image/', 'raw_image/')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, raw_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return True


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parser.parse_args()
    detection_threshold = 0.0
    # object detector
    object_detector = MtcnnArcFaceClassifier(args)

    # detection result handlers
    result_handlers = []

    # [db result handler] update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    # [Line result handler]setup your line channel token and audience
    channel_access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    # denoise filter
    denoise_filters = []
    denoise_filter = FeedbackBboxDeNoiseFilter(
        database, detection_threshold=detection_threshold)
    denoise_filters.append(denoise_filter)

    flask_wrapper = BboxObjectDetectionFlaskWrapper(
        object_detector, bbox_sqlite_handler, result_handlers,
        database=database, drawn_image_dir=args.drawn_image_dir,
        detection_result_filters=denoise_filters)

    params = {'host': args.detector_host, 'port': args.detector_port, 'threaded': False}
    flask_wrapper.app.run(**params)
