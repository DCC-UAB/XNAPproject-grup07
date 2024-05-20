# XNAP-Painting Processing 
In this project we are working with a set of images representing paintings and we have some information related to this artisitic works. There are many goals we can try to achieve, such as detecting the painter or the sytle, we can even predit two characteristics at the same time. 

We are based on a starting point, which works with Keras and works with pretrained weights of Imagenet.

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

<p align="center">
    <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_flip0.jpg" alt="Alt text" width="200"/> <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_flip1.jpg" alt="Alt text" width="200"/> <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_flip2.jpg" alt="Alt text" width="200"/> <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_flip3.jpg" alt="Alt text" width="200"/> </div>

- Data Augmentation (Image Rotate): We have done Data Augmentation with all the folder "train_1" which includes 11025 images. We have rotated them of 45, 90 and 180 degrees counterclockwise and saved the rotated version of the image in a new folder called "train_1_rotate". In this way, we have four times the initial number of images .

- Optimizer (SGD): By applying the optimizer SGD instead of Adam with around 80000 images, we have the following evolution of accuracy:

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/ebd4ecb6b481321ae315b75d05784402dd550ecc/ouput/model_accuracy.png)

    In both models we have overfitting, in SGD we reduce the accuracies' gap. Nevertheless, we obtain worse accuracies with the new optimizer. So, we wouldn't consider it.

    If we look the loss graphic, we can say that with SGD learns slower and worse than the optimizer's starting point. Moreover, seeing the high values of SGD loss we understand that the model isn't suitable for our data. 

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/ebd4ecb6b481321ae315b75d05784402dd550ecc/ouput/model_loss.png)

- Optimizer (RMSprop): After applaying the Adam and SGD optimezers, we also tried to apply the optimizer RMSprop with around 80000 images, as a result we have the following evolution of accuracy:


    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/f8232212d00e00508efd644bc216e4bf4f5522a1/ouput/model_accuracy_adam_sgd_rsmprop.png) 

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/f8232212d00e00508efd644bc216e4bf4f5522a1/ouput/model_loss_adam_sgd_rsmprop.png)

    
    In terms of accuracy, Adam and RMSprop seem to be the most effective optimizers, with similar performance considering the values of accuracy and loss. However, in the end, the results obtained with the Adam optimizer are slightly better. As we have seen previosly,SGD optimizer appears to lag behind compared to the other two, as it has significantly lower performance in terms of accuracy and loss.



## Week 2
### Target goals
- Constructing a mosaic of photos of the same artist. 
- Select the final train we want to use.
- Merging the codes of the first week: 
    - have graphics with the three optimizers together 
    - apply data augmentation to the different optimizers.

### Planning
- Mosaics -> Mercè De la Torre
- Unifying optimizers with data augmentation -> Marta Monsó
- Selection of our final train -> Ariadna
- Applying wandb


### Results
- Mosaics: We've tried the compositions of different images with a very reduced dataset, exactly with 631 files for train and 391 samples for the test. The accuracies using the TEST directory are very low, but we expect to increase them using much more samples:
    - Adam: test accuracy on new images for TRAINED artists = 0.5063
    - Adam: test accuracy on new images for UNEXPECTED artitsts =  0

    - SGD: test accuracy on new images for TRAINED artists = 0.4561
    - SGD: test accuracy on new images for UNEXPECTED artitsts =  0

    So, we've created a directory named 'compositions' to keep the mosaics, for now we have the ones resulting of the Adam's predictions with the reduced dataset. For instance, paints by Pablo Picasso would be like:

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/7b87d343c7eb66e18fad638b09197006617e3c05/compostions/predictions_631_images/composition_Pablo%20Picasso.png)

    Or John Singer Sargent, where we can see some simalirities between the works. Nevertheless, the last picture is quite different from others.

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/7b87d343c7eb66e18fad638b09197006617e3c05/compostions/predictions_631_images/composition_John%20Singer%20Sargent.png)

- Final train: We had a problem with the capacity and the volume of our train set, so we have implemented a code where we select the 10 artist which have the most paintings on the original train. Once we know the artists, we select all their paintings (from the 9 folders of train) and we save only this ones. 
    Now that we have these images, we do Data Augmentation we did last week so we'll have much more paintings in our training set.
    
    And finally we have to confirm that the volume of this new training set is correct for our situation.

- Trying wandb: With wandb we can create graphics and save them in a workspace in wandb. So, we have interactive graphics and "play" with layers of different executions. We've tried it with Adam with 631 images in training set. An exemple of the overwiew: 

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/wandb_dashboard_631_images.png)


## Week 3
### Target goals
- Locating in the training dataset what can give us overfitting and see if it's unbalanced.
- Improving our model.
- Trying different models.

### Planning
- Balancing our dataset -> Ariadna
- Improving our model -> Marta, Mercè and Daria
    - Finetuning the best optimizer -> Mercè
- Try deeper models not used, Resnet 18 and Resnet101 -> Daria

### Results
- Finetuning the best optimizer, which is Adam.
    - Freezing all layers.
    - Unfreezing the last layer.
    - Unfreezing all layers.
    We can see the best practice for our images in the graphic below.



