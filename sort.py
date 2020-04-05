"""
Mattia Amico
A1 - Tracking and detecting people
"""
from __future__ import print_function

from numba import jit
import os.path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
import cv2
from filterpy.kalman import KalmanFilter
from video import generate_video_from_images

@jit
def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
    
  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]
    
  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):  #NB original VALUE iou_threshold = 0.3
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)
  
  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []

  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)
  
  print("UNMATCHED DETECTIONS: ")
  print(np.array(unmatched_detections))
  print("UNMATCHED TRACKERS: ")
  print(np.array(unmatched_trackers))

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
  def __init__(self,max_age=5,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:]) 
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='detection with YOLO, tracking with SORT')
  parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
  #parser.add_argument("-i", "--inputImagesFolder", type=str, default="images/im1" help="path to input images folder")
  parser.add_argument("-d", "--outputFilesPath", type=str, default="output", help="path to detection and tracking output txt files")	
  parser.add_argument("-y", "--yolo", type=str, default="yolo-coco", help="base path to YOLO directory")
  parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
  parser.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when parserplying non-maxima suppression")

  args = parser.parse_args()
  return args

def generate_YOLO_detection(img_dir):
  args = parse_args()
  # load the COCO class labels our YOLO model was trained on
  labelsPath = os.path.sep.join([args.yolo, "coco.names"])
  LABELS = open(labelsPath).read().strip().split("\n")
  
  if(display):
    if not os.path.exists('images/im1'):
      print('\n\tERROR: im1 link not found!\n\n    Create a symbolic link to the m1 dir\n')
      exit()

  # initialize a list of colors to represent each possible class label
  np.random.seed(42)
  COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

  # derive the paths to the YOLO weights and model configuration
  weightsPath = os.path.sep.join([args.yolo, "yolov3.weights"])
  configPath = os.path.sep.join([args.yolo, "yolov3.cfg"])

  # load our YOLO object detector trained on COCO dataset (80 classes)
  print("[INFO] loading YOLO from disk...")
  net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

  # determine only the *output* layer names that we need from YOLO
  ln = net.getLayerNames()
  ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

  #img_dir = args.inputImagesFolder

  data_path = os.path.join(img_dir,'*.jpg')
  imagesPath = [file for file in glob.glob(data_path)]
  imagesPath.sort()
  #print(imagesPath)

  images = [cv2.imread(file) for file in imagesPath]
  total = 795

  #initialize video writer and detection output file
  writer = None
  detectionFilePath = 'output/detection.txt'
  detection_out_file = open(detectionFilePath,"w") 
  frameCounter = 1

  for image in images:
    
    # load our input image and grab its spatial dimensions
    #image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = [] #The detected objects class label
    boxesWithConfidence = []

    # loop over each of the layer outputs
    for output in layerOutputs:
      # loop over each of the detections
      for detection in output:
        
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        if classID == 0:
          confidence = scores[classID]

          # filter out weak predictions by ensuring the detected
          # probability is greater than the minimum probability
          if confidence > args.confidence:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            #(centerX, centerY, width, height) = box.astype("float")
            
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
          
            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            #boxes.append([x, y, width, height])
            
            confidences.append(float(confidence))
            classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes keeping only the most confident ones.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.threshold)

    #draw the boxes and class text on the image

    # ensure at least one detection exists
    if len(idxs) > 0:
      # loop over the indexes we are keeping
      for i in idxs.flatten():
      # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
      
        boxWithConfidence = [boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][3], confidences[i]] #TD check confidence
        boxesWithConfidence.append(boxWithConfidence)
        
        print('%d,%d,%d,%.2f,%.2f,%.4f'%(frameCounter,x,y,w,h,confidences[i]),file=detection_out_file)

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        #text: string label for detected bounding boxes
        text = "{}: {:.4f}".format(LABELS[classIDs[i]] , confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)	

    if writer is None:
      # initialize our video writer
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      writer = cv2.VideoWriter("output/detection.avi", fourcc, 30, (image.shape[1], image.shape[0]), True)
    if total > 0:
      elap = (end - start)
      print("[INFO] single frame took {:.4f} seconds".format(elap))
      print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))	

    #write the output frame to disk
    writer.write(image)
    print("frame written")
    frameCounter += 1

  # release the file pointers
  print("[INFO] cleaning up...")
  writer.release()
  detection_out_file.close()

if __name__ == '__main__':
  args = parse_args()
  display = args.display
  total_time = 0.0
  total_frames = 0 
  colours = np.random.rand(32,3) #used only for display
  
  img_dir="images/im1"
  if not os.path.exists('output'):
    os.makedirs('output')

  generate_YOLO_detection(img_dir)

  if(display):
    plt.ion()
    fig = plt.figure() 
  
  data_path = os.path.join(img_dir,'*.jpg')
  imagesPath = [file for file in glob.glob(data_path)]
  imagesPath.sort()
  images = [cv2.imread(file) for file in imagesPath]
  total = 795

  frameCounter = 1
    
  mot_tracker = Sort() #create instance of the SORT tracker
  detectionFilePath = os.path.join(args.outputFilesPath,'detection.txt')

  #load YOLO detections
  seq_dets = np.loadtxt(detectionFilePath,delimiter=',')
  detection_file = open(detectionFilePath,'w') 

  print(seq_dets)
  print("LEN int(seq_dets[:,0].max()): " + str(int(seq_dets[:,0].max())) )

  with open(os.path.join(args.outputFilesPath,'tracking.txt'),'w') as out_file:
  
    for frame in range(int(seq_dets[:,0].max())): #range(0,795)
     
      frame += 1 #detection and frame numbers begin at 1
      print("Processing frame " + str(frame))

      #retrieve values at indexes for all detections at index==frame
      dets = seq_dets[ seq_dets[:,0]==frame,1:6 ]  #NB: nested list2
      print("dets for frame" + str(frame) + ": ")
      print(dets)

      #convert [x1,y1,w,h] to [x1,y1,x2,y2] ≈ [:,:,x2,y2] = [:,:,w+x1,h+y1]
      dets[:,2:4] += dets[:,0:2]
      print("dets for frame, after [x1,y1,x2,y2] conversion" + str(frame) + ": ")
      print(dets)
    
      total_frames += 1

      if(display):
        ax1 = fig.add_subplot(111, aspect='equal')
        fn = os.path.join(img_dir,'%06d.jpg'%(frame))
        #fn ='images/img1/%06d.jpg'%(frame)
        im =io.imread(fn)
        ax1.imshow(im)
        plt.title('Tracked Targets')
        print("try to print")

      start_time = time.time()
      trackers = mot_tracker.update(dets)
      cycle_time = time.time() - start_time
      total_time += cycle_time

      for d in trackers:
        w = d[2]-d[0]
        h = d[3]-d[1]

        cX = d[0] + w/2
        cY = d[1] + h/2

        print('%d,%d,%.2f,%.2f'%(frame,d[4],cX,cY),file=out_file)

        if(display):
          d = d.astype(np.int32)       
          ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=2,ec=colours[d[4]%32,:]))
          # x_center, y_center of tracker
          ax1.add_patch(patches.Circle((cX,cY),3,fill=False,ec=colours[d[4]%32,:]))
          #label: string ID for detected BBs
          ax1.annotate('id = %d' % (d[4]), xy=(d[0], d[1]), xytext=(d[0], d[1]))
          ax1.set_adjustable('box')
          
      if(display):
        fig.canvas.flush_events()
        plt.draw()
        plt.savefig("output/img_{:03}.png".format(frame))
        ax1.cla()

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  if(display):
    print("Note: to get real runtime results run without the option: --display")

  no_conf_dets = seq_dets[:,:5]
  print("DETECTION LENGHT:")
  print(no_conf_dets)

  for det in no_conf_dets:
    print('%d,%d,%d,%.2f,%.2f'%(det[0],det[1],det[2],det[3],det[4]),file=detection_file)

  generate_video_from_images(args.outputFilesPath)
  
  tmp_images_path = os.path.join(args.outputFilesPath,'*.png')
  imagesToDeletePath = [file for file in glob.glob(tmp_images_path)]
  for file in imagesToDeletePath:
    os.remove(file)

  # release the file pointers
  print("[INFO] cleaning up pointers...")
  
  detection_file.close()
  out_file.close()
  
