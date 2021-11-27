<!--
 * @Descripttion: 
 * @version: 
 * @Author: LiQiang
 * @Date: 2021-11-25 16:06:15
 * @LastEditTime: 2021-11-27 09:20:35
-->
# yolo-fastestv2-opencv
使用OpenCV部署Yolo-FastestV2，包含C++和Python两种版本


# 模型转换
```
python pytorch2onnx.py --data data/coco.data --weights modelzoo/coco2017-0.241078ap-model.pth --output yolo-fastestv2-opencv.onnx
```

opencv-python==4.5.3.56

# Test
```
python main-opencv.py
```


# Reference
https://github.com/hpc203/yolo-fastestv2-opencv

https://github.com/dog-qiuqiu/Yolo-FastestV2
