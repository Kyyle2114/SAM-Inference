# SAM-Inference

Using Segment Anything model in Windows (for cv2.imshow())

## Installation
- ```pip install opencv-python numpy torch matplotlib```
- CUDA is recommended
- [Download](https://github.com/facebookresearch/segment-anything#model-checkpoints) model checkpoint - vit_h

## How to use 
- ```python sam_inference.py```
- Input Image path : imgs/bird.jpg or imgs/car.jpg or your_image_path
- Select Segmentation Mode (Single / Multiple)
    - Single object segmentation mode supports multiple points and single bbox.
    - Multiple object segmentation mode supports multiple bboxes.
- Press 'p' : Points mode
    - Positive point : left mouse click
    - Negative point : right mouse click
- Press 'b' : Bbox mode
    - Mouse dragging
- Press 'r' : Reset all prompts
- Press 'i' : Model inference 

## Examples 

You can also see an example code in SAM.ipynb.

## Hardware

Works fine on GTX 1660 Ti.