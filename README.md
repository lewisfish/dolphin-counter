# Dolphin-counter

This project aims to use machine learning techniques to count the amount of dolphins in aerial drone footage.

## Usage

python dolphins.py [-h] [-f FILE] [-d{d}]

- -h, shows the help screen.
- -f, Path to single image to be analysed.
- -d, Display debug info. Has differnet levels of debug information, e.g -dd shows more debug info than -d

## Installation

If you are using anaconda to manage your python enviroment then use the following to install all dependancies

   - conda env create -f enviroment.yml

Or if you use pip
   - pip install --user --requirment requirments.txt

## Requirments

  - opencv
  - numpy
  - scikit-image
  - matplotlib

## Usage


## ToDo
 - [x] Generate candidates dolphin objects using thresholding, watershedding etc
 - [ ] Refine this
 - [ ] Add Machine Learning
 - [ ] More
