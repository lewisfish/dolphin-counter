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

## Requirments
 
  - astropy 4.0+
  - opencv 4.2.0+
  - matplotlib 3.1.2+
  - numpy 1.17.3+
  - photutils 0.7.2+
  - scikit-image 0.16.2+

## ToDo
 - [x] Generate candidates dolphin objects using thresholding, watershedding etc
 - [ ] Refine this
 - [ ] Add Machine Learning
 - [ ] More
