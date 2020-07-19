#!/bin/bash
python object_detection_webserver.py --prototxt MobileNetSSD_deploy.prototxt \
    --model MobileNetSSD_deploy.caffemodel --ip 192.168.1.20 --port 8000
