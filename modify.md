# Installation
## 1. Follow caffe installation

## 2. Modify Make.Config as py-faster-rcnn
But, need to modify as below:
```text
# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

## 3. Need modify for python3
print, ...

## 4. lib/fast_rcnn/train.py
Add "import google.protobuf.text_format"

## 5. float vs int index 
--> line 124, 125
py-faster-rcnn/lib/rpn/proposal_target_layer.py
--> int()

## 6. For numpy
```text
lib/roi_data_layer/minibatch.py: 
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image) -->
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```
 
 ```text
 lib/datasets/ds_utils.py
hashes = np.round(boxes * scale).dot(v)-->
hashes = np.round(boxes * scale).dot(v).astype(np.int)
```

```text
lib/fast_rcnn/test.py
hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v) -->
hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v).astype(np.int)
```
```text
lib/rpn/proposal_target_layer.py
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image) -->
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```
