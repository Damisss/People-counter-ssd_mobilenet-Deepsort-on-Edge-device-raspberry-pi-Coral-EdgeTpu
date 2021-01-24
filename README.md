# People-counter-ssdmobilenet-Deepsort-on-Edge-device-raspberry-pi-Coral-EdgeTpu
<img src="/result.gif" width="350" height="300"/>

This repository attempts to depict how to implement object tracker on resource-constrained hardware (raspberry-pi Coral Edge TPU usb Accelerator). For instance it can be use to track people entering/exiting a store. It demostrates how to implementates and perform realtime tracking with Tensorflow using a SSD model trained v2 pretrained model.The object tracking is based on the Simple Online and Realtime Tracking with a Deep Association Metric [Deep SORT](https://github.com/nwojke/deep_sort) algorithm.

  - Please follow [Pyimagesearch's tutorial](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator) to setup the rasberry pi for google coral's TPU usb Accelerator.
 
# Inference:

To run real time inference:
python object_tracker.py.
Note that --input, --confidence,  --cosine_distance, --nms_thresholdare, --classes, --direction_mode and --output are optional. One can ajust any of those commands line to their need. 
  - --input is the path to input video
  - --confidence is the  minimum proba to filter weak detections
  - --cosine_distance is the cosine distance (deep sort param)
  - --nms_thresholdare is non-maximum suppression threshold (deep sort param)
  - --classes is the classes to be detected and tracked (e.g. person)
  - --direction_mode is the direction mode of the objects (e.g vertical/ horizontal)
  - --output is the path to output vide

# References

  - https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/
  - https://coral.ai/docs/
  - https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/
  - https://github.com/omarabid59/TensorflowDeepSortTracking
