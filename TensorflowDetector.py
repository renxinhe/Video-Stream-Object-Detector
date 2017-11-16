import os, sys, re
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
sys.path.append('TensorflowResearch/')

from AbstractDetector import AbstractDetector
from utils import label_map_util
from utils import visualization_utils as vis_util

if not re.match(r'.*1\.4\.0.*', tf.__version__):
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!', tf.__version__)

"""
    Based on https://github.com/tensorflow/models/tree/master/research/object_detection

    MODEL_NAME can be any model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        e.g.: ssd_mobilenet_v1_coco_11_06_2017
"""
class TensorflowDetector(AbstractDetector):
    def __init__(self, architechture, model_name, stream_url=None):
        super(TensorflowDetector, self).__init__(architechture, stream_url=stream_url)

        # What model to download.
        MODEL_NAME = model_name
        MODEL_FILE = 'TensorflowResearch/models/' + MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = 'TensorflowResearch/models/' + MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('TensorflowResearch/data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

        # Download model
        if not os.path.isfile(MODEL_FILE):
            try:
                opener = urllib.request.URLopener()
                opener.retrieve(DOWNLOAD_BASE + MODEL_NAME + '.tar.gz', MODEL_FILE)
            except IOError as e:
                print 'Fetching weight at %s failed.' % (DOWNLOAD_BASE + MODEL_FILE)
                sys.exc_info()

        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd() + '/TensorflowResearch/models/')

        # Load frozen TF model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Setup Tensorflow Session
        with self.detection_graph.as_default():
            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.

    def process_image(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        boxes, scores, classes, num = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        return boxes, scores, classes, num

    def drawBoundingBox(self, frame):
        image_np = frame
        boxes, scores, classes, num = self.process_image(image_np)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            min_score_thresh=0.5,
            line_thickness=4)
        return image_np

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
            