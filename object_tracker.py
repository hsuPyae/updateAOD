import os

# comment out below line to enable tensorflow logging outputs
from trackers.detection import Detection
from trackers.object_tracker import ObjectTracker
from trackers.person_tracker import PersonTracker

from tools import generate_detections as gdet
from tracker_utils.iou_matching import iou_cost

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import pafy

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from tracker_utils import preprocessing

flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_string('output', 'res.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.2, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

weights = './checkpoints/yolov4-608'

# USED_CLASSES = [27, 28, 31, 33, 84]
# GROUPS_OF_CLASSES = [[27, 31, 33], [28], [84]]
# PERSON_CLASS = 1
# allowed_classes = ['person', 'suitcase', 'book', 'backpack', 'umbrella', 'handbag']
GROUPS_OF_CLASSES = [['suitcase', 'backpack', 'handbag'], ['book'], ['umbrella']]


def get_video_frame():
    x = input("Enter 1 for youtube, 2 for video,3 for webcam : ")
    if x == '1':
        text = input("Enter your link: ")
        url = text
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")

        cap = cv2.VideoCapture()  # Youtube
        cap.open(best.url)

    elif x == '2':
        x = input("Enter your video: Type Format ..avi or .mp4 : ")
        cap = cv2.VideoCapture(x)  # Import Video File

    elif x == '3':
        cap = cv2.VideoCapture(0)  # webcam
        cap.set(3, 1000)
        cap.set(4, 1000)
        print(cap.get(3))
        print(cap.get(4))
    else:
        raise Exception("Your input is wrong")
    return cap, x


def update_objects(detections, person_tracker, object_trackers, time):
    p_detections = []
    o_detections = [[] for _ in range(len(GROUPS_OF_CLASSES))]
    for detection in detections:
        if detection.class_id == 'person':
            p_detections.append(detection)
        for i, group in enumerate(GROUPS_OF_CLASSES):
            if i == 2 and detection.class_id == 'book':
                print('BOOK!!!')
            if detection.class_id in group:
                o_detections[i].append(detection)

    person_tracker.predict()
    person_tracker.update(p_detections)

    for i, objects_tracker in enumerate(object_trackers):
        objects_tracker.predict()
        objects_tracker.update(o_detections[i], time)


def check_whether_object_is_abandoned_for_a_long_time(object_trackers, cur_time, prev_abandoned_tracks):
    abandoned_tracks = []
    for object_tracker in object_trackers:
        for track in object_tracker.tracks:
            if track.is_abandoned():  # track.is_abandoned() track.time_since_update < 3
                # tlbr = track.to_tlbr().copy()
                # tlbr[:2] = tlbr[:2:-1]
                # tlbr[:2] = tlbr[:2:-1]
                # abandoned_objects['detection_boxes'].append(track.to_tlbr())
                # abandoned_objects['detection_scores'].append(track.confidence)
                # abandoned_objects['detection_classes'].append(track.class_name)
                # abandoned_objects['owner'].append(track.owner)
                #
                # abandoned_objects['time'].append(cur_time)
                abandoned_tracks.append(track)

    abandoned_tracks = filter(lambda x: (cur_time - x.abandoned_time) <= 2., abandoned_tracks)
    sorted_abandoned_tracks = sorted(abandoned_tracks, key=lambda x: -x.confidence)
    abandoned_tracks = [] #list(filter(lambda x: (cur_time - x.abandoned_time) <= 2., prev_abandoned_tracks))
    for obj in sorted_abandoned_tracks:
        if len(abandoned_tracks) == 0:
            abandoned_tracks.append(obj)
        elif iou_cost([obj], abandoned_tracks).max() > 0.2:
            if obj not in abandoned_tracks:
                abandoned_tracks.append(obj)

    return abandoned_tracks


def main(_argv):
    # begin video capture
    vid, x = get_video_frame()
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # initialize tracker
    min_score_thresh_for_peoples = .10
    min_score_thresh = .20
    maxDisappeared = 120
    skip_frames = 3
    person_tracker = PersonTracker(max_distance=0.5, max_age=300 // skip_frames,
                                   min_score_thresh=min_score_thresh_for_peoples)
    objects_trackers = [
        ObjectTracker(person_tracker, max_distance=0.1, max_age=maxDisappeared // skip_frames,
                      min_score_thresh=min_score_thresh,
                      n_init=5) for _
        in GROUPS_OF_CLASSES]

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    out = None

    # get video ready to save locally if flag is set
    video_fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, video_fps, (width, height))
    frame_num = 0
    abandoned_tracks = list()
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        if frame_num == 1321:
            print()
        if frame_num == 661:
            print()
        if frame_num == 1801:
            print()
        if frame_num % skip_frames == 1:
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            book_pred_conf = pred_conf.numpy()
            book_pred_conf[:, :, :73] = 0.
            book_pred_conf[:, :, 74:] = 0.
            pred_conf = pred_conf.numpy()
            pred_conf[:, :, 73] = 0.
            pred_conf = tf.convert_to_tensor(pred_conf)
            book_pred_conf = tf.convert_to_tensor(book_pred_conf)
            book_boxes = boxes
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=40,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            book_boxes, book_scores, book_classes, book_valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(book_boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    book_pred_conf, (tf.shape(book_pred_conf)[0], -1, tf.shape(book_pred_conf)[-1])),
                max_output_size_per_class=10,
                max_total_size=10,
                iou_threshold=FLAGS.iou,
                score_threshold=0.01
            )
            # print('valid book detection', book_valid_detections.shape)
            # print('valid book detection', book_valid_detections)
            # print('valid detection', valid_detections)
            # print('book_scores', book_scores)
            # print('scores', scores)

            boxes = tf.concat([boxes[:, :valid_detections[0]], book_boxes], axis=1)
            scores = tf.concat([scores[:, :valid_detections[0]], book_scores], axis=1)
            classes = tf.concat([classes[:, :valid_detections[0]], book_classes], axis=1)
            # book_valid_detections = book_valid_detections + 40
            valid_detections = valid_detections + book_valid_detections

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['person', 'suitcase', 'book', 'backpack', 'umbrella', 'handbag']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            bboxes[:, 0] /= width
            bboxes[:, 1] /= height
            bboxes[:, 2] /= width
            bboxes[:, 3] /= height
            if np.any(bboxes > 1.0) or np.any(bboxes < 0.):
                raise Exception
            # bboxes /= norm_boxes

            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(bboxes, scores, names, features)]

            # initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            cur_time = 1. * frame_num / video_fps
            update_objects(detections, person_tracker, objects_trackers, time=cur_time)
            abandoned_tracks = check_whether_object_is_abandoned_for_a_long_time(objects_trackers, cur_time,
                                                                                 abandoned_tracks)

        # update tracks
        # for track in person_tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     bbox = track.to_tlbr()
        #     bbox[0] *= width
        #     bbox[1] *= height
        #     bbox[2] *= width
        #     bbox[3] *= height
        #     # bbox *= norm_boxes
        #     class_name = track.get_class()
        #
        #     # draw bbox on screen
        #     color = colors[int(track.track_id) % len(colors)]
        #     color = [i * 255 for i in color]
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
        #                   (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
        #     cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
        #                 (255, 255, 255), 2)
        #
        #     # if enable info flag then print details about each track
        #     if FLAGS.info:
        #         print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
        #                                                                                             class_name, (
        #                                                                                                 int(bbox[0]),
        #                                                                                                 int(bbox[1]),
        #                                                                                                 int(bbox[2]),
        #                                                                                                 int(bbox[3]))))

        # update tracks
        for track in abandoned_tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            bbox[0] *= width
            bbox[1] *= height
            bbox[2] *= width
            bbox[3] *= height
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),
                          color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0,
                        0.75,
                        (255, 255, 255), 2)

            # if enable info flag then print details about each track
            print("Abandoned object: {}, Class: {}, owner: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id),
                track.owner,
                class_name, (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
