#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python clothes_test_ori.py blouse ;
CUDA_VISIBLE_DEVICES=0 python clothes_test_ori.py dress;
CUDA_VISIBLE_DEVICES=0 python clothes_test_ori.py outwear;
CUDA_VISIBLE_DEVICES=0 python clothes_test_ori.py skirt;
CUDA_VISIBLE_DEVICES=0 python clothes_test_ori.py trousers;
