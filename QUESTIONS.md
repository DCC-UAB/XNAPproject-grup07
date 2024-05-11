# XNAP-Painting Processing 
In this project we are working with a set of images representing paintings and we have some information related to this artisitic works. There are many goals we can try to achieve, such as detecting the painter or the sytle, we can even predit two characteristics at the same time. 

We are based on a starting point, which works with Keras.

## Week 1
### Target goals
- Detect the painter of the painting.
- Try two more different optimizers (RMSProp, SGD) and compare their accuracy.
- Do some Data Augmentation and see if the accuracy improves.

### Planning
We divided our target goals in 4 mini-goals so each one of us could improve one:
- Data Augmentation (Image flip) -> Ariadna Lucero
- Data Augmentation (Image Rotation) -> Daria Buiuc
- Optimizer (RMSProp) -> Marta Monsó
- Optimizer (SGD) -> Marcè De la Torre

### Results
- Data Augmentation (Image flip): We have done Data Augmentation with all the folder "train_1" which includes 11025 images. In this first part we just flip horizontal, vertical and horizontal and vertical, creating then 3 new images for every image on the train folder. Except for "../../train_1\121.jpg" that is is not the same format and it can't read it, we did it for all the others. Then we have a new folder "train_1_flip" with all the new train containing 33072 images. If we use all the images with the model, we have a total of 44097 images for testing at the moment. The next step for this part is to try with the models, which one has better Accuracy and study the different cases.

- Data Augmentation (Image Rotate): We have done Data Augmentation with all the folder "train_1" which includes 11025 images. We have rotate them of 45 degrees counterclockwise and saved the rotated version of the image in a new folder called "train_1_rotate". In this way, we have twice the initial image .

- Optimizer (SGD): By applying the optimizer SGD instead of Adam we have the following evolution of accuracy:
![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/ebd4ecb6b481321ae315b75d05784402dd550ecc/ouput/model_accuracy.png)

In both models we have overfitting, in SGD we reduce the accuracies' gap. Nevertheless, we obtain worse accuracies with the new optimizer. So, we wouldn't consider it.

If we look the loss graphic, we can say that with SGD learns slower and worse than the optimizer's starting point. Moreover, seeing the high values of SGD loss we understand that the model isn't suitable for our data. \
![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/ebd4ecb6b481321ae315b75d05784402dd550ecc/ouput/model_loss.png)


## Week 2


## Week 3