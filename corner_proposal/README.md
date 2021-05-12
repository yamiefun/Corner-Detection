# Corner Proposal
## Introduction
This tool can provided candidate corners for CCTV calibration. Users only need to choose the corner which is the nearest to the real corner pixel.

The input should be a CCTV camera scene. This tool will first run Harris corner detector to detect all corners in the image, and filter most of them out with indoor segmentation information. The remaining corners should only appear in between walls and floor.

The output image will be generate in `output` folder. 
You could customize the code for your own applications. For example, return all candidate corners coordinates to the frontend for rendering.

## Installation
This tool is based on [Indoor Segmentation](https://github.com/hellochick/Indoor-segmentation). Please take a look at the [README](https://github.com/hellochick/Indoor-segmentation/blob/master/README.md) for more details. 

## How to Run
Put your input image in `input` folder and  simply run 
```
python3 main.py
```
