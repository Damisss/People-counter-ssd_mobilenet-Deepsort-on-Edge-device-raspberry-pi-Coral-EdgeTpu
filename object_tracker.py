#Usage python object_tracker.py
# optional command line args
# --input \
# --confidence \
# --cosine_distance \
# --nms_threshold \
# --classes \
# --direction_mode \
# --output 
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import cv2
import numpy as np
from multiprocessing import Process, Value, Queue
import datetime
import argparse

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from utils import config
from utils.direction_counter import DirectionCounter
from utils.trackable_object import TrackableOBject
from utils.video_writer import VideoWriter


#arguments parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', help='Path to input video.')
ap.add_argument('-c', '--confidence', default=.7, type=float,  help='Minimum proba to filter weak detections')
ap.add_argument('-d', '--cosine_distance', default=.6, type=float,  help='Cosine distance')
ap.add_argument('-t', '--nms_threshold', default=.8, type=float,  help='Non-maximum suppression threshold')
ap.add_argument('-cl', '--classes', default='person', type=str,  help='Classes to be detected.')
ap.add_argument('-dm', '--direction_mode', default='vertical', type=str,  help='Direction mode')
ap.add_argument('-o', '--output', default='output/result.avi',  help='Path to output video')
args = vars(ap.parse_args())

class ObjectTracker ():
  @staticmethod
  def start():
    try:
      # labels 
      LABELS = config.LABELS
      #classes to be detected
      LIST_CLASSES = args['classes'].split(',')
      COLORS = np.random.uniform(0, 255, size=(len(LIST_CLASSES), 3))
      CLASS_COLORS = {}
      #load model form disk
      net = DetectionEngine(config.SSD_MOBILENET_V2)
      #load deepsort model from disk
      encoder = gdet.create_box_encoder(config.DEEP_SORT_MODEL, batch_size=1)
      #instatiate metrics
      metric = nn_matching.NearestNeighborDistanceMetric('cosine', args['cosine_distance'], None)
      # instatiate tracker
      tracker = Tracker(metric)
      

      if not args.get('input', False):
        cap = cv2.VideoCapture(1)
      else:
        cap = cv2.VideoCapture(args['input'])
      
      #instatiate video writer
      videoWriter = VideoWriter(args['output'], cap.get(cv2.CAP_PROP_FPS))
      videoWriterProcess = None
      # initialize frame height and width
      H, W = None, None
      trackableObjects ={}
      directionInfo = []
      detections = []
      
      totalFrames = 0
      up = 0
      down = 0
      startTime = datetime.datetime.now()
      while cap.isOpened():
        ret, frame = cap.read()
        # stop while loop if is there is no frame available 
        if not ret:
          break
        
        #stop while loop if q is pressed
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
          break

        if H is None or W is None:
          H, W = frame.shape[:2]
          dc = DirectionCounter(args['direction_mode'], H, W)
        
        if args['direction_mode'] == 'vertical':
          cv2.line(frame, (0, H//2), (W, H//2), (0, 255, 0), 2)
        else:
          cv2.line(frame, (W//2, 0), (W//2 , H), (0, 255, 0), 2)
        #initialize process if it is not yet initialize.
        if args['output'] and videoWriterProcess is None:
          writeVideo = Value('i', 1)
          frameQueue = Queue()
          videoWriterProcess = Process(target=videoWriter.start, args=(writeVideo, frameQueue, W, H))
          videoWriterProcess.start()
          
        #Data preprocessing
        inFrame = cv2.resize(frame, (300, 300))
        inFrame = Image.fromarray(inFrame)
        results = net.detect_with_image(inFrame, threshold=args['confidence'])
        
        #
        names = []
        bboxes = []
        scores = []

        #loop over detection results
        for r in results:
          # label index
          label_id = int(r.label_id + 1)
          # check if detected object is in the list of the object we are interrested in.
          if label_id in LABELS.keys() and LABELS[label_id] in LIST_CLASSES:
            #grab label of detected object
            label_name = LABELS[label_id]
            # scale bounding box
            bbox = r.bounding_box.flatten() * np.array([W, H, W, H])
            # add label to names list
            names.append(label_name)
            # add object bounding box to bboxes
            bboxes.append(bbox)
            # add detection confidence to scores list
            scores.append(r.score)
            # class color
            CLASS_COLORS[label_name] = COLORS[r.label_id]
         
        names = np.array(names)
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        #grab feartures from frame
        features = encoder(frame, bboxes)
        # perform deep sort detection
        detections = [Detection(bbox, score,class_name,  feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        # grab boxes, scores, and classes_name from deep sort detections
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        #perform non-maxima suppression on deep sort detections
        indices = preprocessing.non_max_suppression(boxs, args['nms_threshold'], scores)
        detections = [detections[i] for i in indices]
        # update tracker
        tracker.predict()
        tracker.update(detections)
        # loop over tracked objects
        for track in tracker.tracks:
          
          if not track.is_confirmed() or track.time_since_update >1:
            continue
          # grab object bounding box class_name
          bbox = track.to_tlbr()
          class_name= track.get_class()
          to = trackableObjects.get(track.track_id, None)
          # calculate centroid of each box
          cx, cy = int((bbox[0] + bbox[2])/2.0), int((bbox[1] + bbox[3])/2.0)
          centroid = (cx, cy)
          # perform counting 
          if to is None:
            to = TrackableOBject(track.track_id, centroid)
            
          else :
            dc.find_direction(to, centroid)
            to.centroids.append(centroid)
            if not to.counted:
              directionInfo = dc.count_object(to, centroid)
          # grab direction counts
          up = directionInfo[0][1] if len(directionInfo) else  0
          down = directionInfo[1][1] if len(directionInfo) else 0
          trackableObjects[track.track_id] = to 
          
          #draw bbox and track id on frame 
          cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), CLASS_COLORS[class_name], 2)
          y = int(bbox[1]) - 15 if int(bbox[1]) - 15 > 15 else int(bbox[1]) + 15
          cv2.putText(frame, class_name + ' ' + str(track.track_id), (int(bbox[0]), y), 0, 0.5, CLASS_COLORS[class_name], 2)
          
        # Draw direction count on bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Up:' + ' ' + str(up), (0, H - 40), font, 0.5, (255, 0, 255), 2)
        cv2.putText(frame, 'Down:' + ' ' + str(down), (0, H -15), font, 0.5, (255, 0, 255), 2)
        elips = (datetime.datetime.now() - startTime).total_seconds()
        totalFrames += 1
        fps = totalFrames / elips
        text = 'Average FPS: {:.2f}%'.format(fps)
        cv2.putText(frame, text, (10, 20), font, 0.5, (255, 0, 255), 2)
        # put frame in queue
        if videoWriterProcess is not None:
          frameQueue.put(frame)
        cv2.imshow('track', frame)
        # stop writting process
      if videoWriterProcess is not None:
        writeVideo.value = 0
        videoWriterProcess.join()
        
      cap.release()
      cv2.destroyAllWindows()

    except Exception as e:
      raise e


if __name__ == '__main__':
  ObjectTracker.start()