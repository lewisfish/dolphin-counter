# Dolphin-counter

This project aims to use machine learning techniques to count the amount of dolphins in aerial drone footage.

## Usage

python dolphins.py [-h] [-f FILE] [-d{d}] [-np]

- -h, shows the help screen.
- -f, Path to single image to be analysed.
- -d, Display debug info. Has differnet levels of debug information, e.g -dd shows more debug info than -d.
- -np, Supress default plot output.

## Installation

If you are using anaconda to manage your python enviroment then use the following to install all dependancies

   - conda env create -f enviroment.yml

## Requirments
 
  - opencv
  - numpy
  - scikit-image=0.16.2
  - matplotlib
  - pytesseract
  - python=3.7
  - tensorflow
  - pandas


## ToDo
 - [x] Generate candidates dolphin objects using thresholding, watershedding etc
 - [ ] Refine this
 - [ ] Add Machine Learning
 - [ ] More
