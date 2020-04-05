The task of this assignment is to detect and track people in a video taken from the MOT challenge

- **Tracking**: for each pedestrian in the scene, provide the trajectory of the center of the bounding box. As a validation metric, compute the average displacement error between the center of your bounding box and the the one computed on the ground truth data (ground truth provides only top left corner, width and height of each bounding box).

- **Detection** : for each pedestrian in the scene, provide the detected bounding box using a pedestrian detector.

- - - - 

This code has been tested on Mac OSX 10.14 with Python 3.7 (anaconda)

### Dependencies:

This code makes use of the following packages
1.  [`scikit-learn`](http://scikit-learn.org/stable/)
2.  [`scikit-image`](http://scikit-image.org/download)
3.   [`FilterPy`](https://github.com/rlabbe/filterpy)
and more others

To install required dependencies run:

```
$ pip install -r requirements.txt
```

### How to run:
Dowload the required YOLO pre-trained mode ( https://drive.google.com/file/d/1FaxA6lHYoKV0SDCBtKD5vqKqRXOL8YjC/view?usp=sharing) inside path/to/A1_code/yolo 

```
$ cd path/to/A1_code
```

the frames of the proposed MOT video are located at ```images/img1```, in order to provide the reference data used.
otherwise to run using an alternative video sequence, create a symbolic link from the external im1 directory, containing the frames to be provided as input, to the local images directory:
```
$ ln -s /path/to/videoSequences/img1 images/img
```

To run the code and generate ```detection.txt``` and ```tracking.txt``` files and videos in output/folder:

```
$ python sort.py --display
```

