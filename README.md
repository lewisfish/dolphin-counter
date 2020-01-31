# :dolphin: Dolphin-counter :dolphin:

This project aims to use machine learning techniques to count the amount of dolphins in aerial drone footage.

![Example of dolphin detection](https://raw.githubusercontent.com/lewisfish/dolphin-counter/master/example.png)

## Usage

python dolphins.py [-h] [-f FILE] [-fo FOLDER] [-d{d}] [-np] [-n NCORES]

- -h, shows the help screen.
- -f, Path to single image to be analysed.
- -fo, Path to folder of images to be analysed.
- -d, Display debug info. Has differnet levels of debug information, e.g -dd shows more debug info than -d.
- -np, Supress default plot output.
- -n, Specify the number of cores to use. Default is 1.

## Installation

If you are using anaconda to manage your python enviroment then use the following to install all dependancies

   - conda env create -f enviroment.yml

## Requirments
 
  - opencv
  - numpy
  - scikit-image=0.16.2
  - matplotlib
  - python=3.7
  - tensorflow

## ToDo
 - [x] Generate candidates dolphin objects using thresholding, watershedding etc
 - [ ] Refine this
 - [ ] Add Machine Learning classification
 - [ ] More
