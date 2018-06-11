#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python TJJMask.py blouse sleeve_length_labels /home/tanghm/Documents/YFF/TJJ/base_fusai/Images/sleevenew/;
CUDA_VISIBLE_DEVICES=1 python TJJMask.py outwear coat_length_labels /home/tanghm/Documents/YFF/TJJ/base_fusai/Images/coatnew/;
CUDA_VISIBLE_DEVICES=1 python TJJMask.py skirt skirt_length_labels /home/tanghm/Documents/YFF/TJJ/base_fusai/Images/skirtnew/;
CUDA_VISIBLE_DEVICES=1 python TJJMask.py trousers pant_length_labels /home/tanghm/Documents/YFF/TJJ/base_fusai/Images/pantnew/;
