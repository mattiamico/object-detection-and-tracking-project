import cv2
import numpy as np
import os

def generate_video_from_images(outputPath):
  #Create a VideoCapture object
  cap = cv2.VideoCapture(os.path.join(outputPath,"img_%03d.png"))
    
  # Check if camera opened successfully
  if (cap.isOpened() == False): 
    print("Unable to read camera feed")
  
  # Default resolutions of the frame are obtained.The default resolutions are system dependent.
  # We convert the resolutions from float to integer.
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  
  # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
  out = cv2.VideoWriter(os.path.join(outputPath,'output.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
  
  while(True):
    ret, frame = cap.read()
  
    if ret == True: 
      
      # Write the frame into the file 'output.avi'
      out.write(frame)
     
    else:
      break 
  
  # When everything done, release the video capture and video write objects
  cap.release()
  out.release()
  
  print("output.avi video generated")
  
  # Closes all the frames
  cv2.destroyAllWindows()