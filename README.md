This code has been tested on Mac OSX 10.14 with Python 3.7 (anaconda)

### Dependencies:

This code makes use of the following packages
1.  [`scikit-learn`](http://scikit-learn.org/stable/)
2.  [`scikit-image`](http://scikit-image.org/download)
3.   [`FilterPy`](https://github.com/rlabbe/filterpy)
and more others

To install required dependencies run:

$ pip install -r requirements.txt

### How to run:
Dowload the required YOLO pre-trained mode ( https://drive.google.com/file/d/1FaxA6lHYoKV0SDCBtKD5vqKqRXOL8YjC/view?usp=sharing) inside path/to/A1_code/yolo 

To run the code and generate txt files and videos:

$ cd path/to/A1_code

Create a symbolic link from the external im1 directory, containing the frames to be provided as input, to the local images directory:

$ ln -s /path/to/videoSequences/img1 images/img1

$ python sort.py --display

