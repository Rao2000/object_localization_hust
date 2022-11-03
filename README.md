## 华中科技大学电子信息与通信学院计算机视觉理论结课作业
* 作者: 廖满文、吴雨暄、饶龙、汪潇翔
* 指导老师: 王兴刚、廖振宇

## Course project
* Object localization using CNN
* Data: http://xinggangw.info/data/tiny_vid.zip
* For every class, the first 150 images are used for training the localizer and the 
rest for testing.
* Evaluation metric: A correct localization means the predict class label is correct and the predict box has an IoU>0.5 with the ground-truth. You should report the average accuracy and show some localization results.
* Step #1: implementing the localization network using direct bbox regression
* Step #2: adding anchors in the localization network
* Extra credit: multiple object localization using your own network


