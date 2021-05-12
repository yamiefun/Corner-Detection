# Corner Refinement
## Introduction
This tool can refine the location of the corners selected by users manually, let user only need to roughly select the corners in the image to accelerate the calibration process.

The inputs are images and a `json` file containing all pixels coordinates selected by users.

The output will be an image with both users selected corners (drawn in red) and corners which are refined (drawn in green). You could customize the code to fit your application. For example, only return the refined pixels coordinates to users.

## Installation

## How to Run
Place your images into `input` folder.
Write users selected corners in `gt.json` and place it in the same directory with `main.py`.
Just simply run
```
python3 main.py
```