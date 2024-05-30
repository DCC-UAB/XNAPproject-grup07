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

<p align="center">
    <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_rotate_180.jpg" alt="Alt text" width="200"/> <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_rotate_45.jpg" alt="Alt text" width="200"/> <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dat_aug_rotate_90.jpg" alt="Alt text" width="200"/> </div>

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
- Create augmentation in Keras -> Marta
- Improving our model -> Marta, Mercè and Daria
    - Finetuning the best optimizer -> Mercè
    - Dropout i regularització --> Marta
- Try deeper models not used, Resnet 18 and Resnet101 -> Daria

### Results
- Until now, we had been working with manually implemented rotation and flip augmentations, but we have seen that we need to use ImageDataGenerator to create augmentations for the images in the training set using Keras. We have added the data augmentation part in the definition of train_data_gen within the setup_generators_with_augmentation function. The augmentations include rotations, shifts, shears, zooms, and horizontal flips. This configuration has been specifically applied to the training set in order to improve the robustness of the model.

We run the code using the train set that contains the images of the 10 artists who have painted the most paintings in the original train set, performing the augmentation part and using the three different optimizers: adam, rmsprop, and sgd. We have executed it using wandb.


ADAM

 ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/augmentation_adam.jpg)

RMSPROP

 ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/augmentation_rms.jpg)

SGD

 ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/augmentation_sgd.jpg)



In the results of the three optimizers, we can observe the same trend, where a good accuracy is achieved for the training set but for the validation set, the accuracy remains low, indicating that we have overfitting to our data.

- To try to reduce overfitting, we added a dropout layer to the model configured with a rate of 0.4 and L2 regularization with a factor of 0.01. However, we can see that the trend has not changed, and we still have overfitting.


![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/dropout.jpg)


- Training our model with 2600 images and augmenting the training set with flipped images, what makes 10400 works to train with. And using 10 artists, we obtain the accuracies and losses graphics below:
    
    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/62d1a2792c7dc5e8f0bdce61a8c9b3a4c62acc78/ouput/model_accuracy_3opt_10400im_10art.png)

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/62d1a2792c7dc5e8f0bdce61a8c9b3a4c62acc78/ouput/model_loss_3opt_10400im_10art.png)

    We still have a big overfitting. Nevertheless, we can extract the "best" optimizer: Adam. We could've said RMSProp, but the Adam's tendency of the test accuracy is upwards, whether the RMSProp increases the overfitting.

    If we look up the accuracies, Adam optimizer works better:
    - **Adam: test accuracy on new images for TRAINED artists 209 / 335 = 0.6239**
    - SGD: test accuracy on new images for TRAINED artists 178 / 335 = 0.5313
    - RMSProp: test accuracy on new images for TRAINED artists 189 / 335 = 0.5642

- Training our model with 13000 images (including rotation and flip) in 20 epochs instead of 5. And using 10 artists, we obtain the accuracies and losses graphics below, extracted fromb wandb:

    First we show our dashboard:

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/dashboard_13000im_10art.png)

    And now we can see the comparasion between training and test.
    
    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/model_accuracies_13000im_10art.png)

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/model_loss_1300im_10art.png)

    We obtain slightly better results. However, we still have a big overfitting. 

    If we look up the accuracies, Adam remains being the optimizer that works better:
    - **Adam: test accuracy on new images for TRAINED artists 233 / 335 = 0.6955**
    - SGD: test accuracy on new images for TRAINED artists 227 / 335 = 0.6776
    - RMSProp: test accuracy on new images for TRAINED artists 231 / 335 = 0.6896

    So, predicting with Adam, we've created mosaics for predicted values of each piece of art. And also, we've composed the mosaic that we should've obtained.

    Mosaic with predicted artists:

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/a3fc250e2ff2da10e97120fea8e5e5da436e57bf/compostions/predictions_13000_images/composition_Gustave%20Dore.png)

    Mosaic with real value:

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/a3fc250e2ff2da10e97120fea8e5e5da436e57bf/compostions/real_values_13000_images/composition_Gustave%20Dore.png)

- Finetuning the best optimizer, which is Adam.
    - Freezing all layers.
    - Unfreezing the last layer.
    - Unfreezing all layers.
    We can see the best practice for our images in the graphic below.

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/model_accuracies_finetuning_13000im_10art.png)

    There's a lot of overfitting in the three ways, but in unfreezing one layer, the last layer, we obtain better results, specially in the test.

    Now we compare the losses:

    ![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/model_losses_finetuning_13000im_10art.png)

    Now we confirm that we have less loss when unfreezing the last layer.



## Week 4
### Target goals
We have decided that working with 30 artists doesn't give us the results we expected. So we will now work with 10 artists again. And if we don't improve the model, at least we will find the problems and try to fix them.
- We want to do a research for overlapping and identify if we have this problem.
- Try to freeze by layers and try different combinations.
- Add relu and also try with model Resnet18.
- Weights from the different classes and observe the changes.
- Do a correct distribution for the classes in train and validation set.
- Compare the results from accuracy into different classes (artists).

### Planning
- Get the 10 artists with the most pictures -> Ariadna
- Compare accuracy from classes -> Daria

### Results
In the first task of separating the dataset into the top 10 artists, we had to consider the actual number of photos we had for each artist because the initial dataframe seemed to have more photos than we actually found. This resulted in the final set of photos being highly unbalanced. Therefore, we decided to select 10 artists with approximately the same number of photos. With these artists, we will conduct the initial tests.

- At the moment, this first 10 artists will be: Camille Pissarro, Vincent van Gogh, Theophile Steinlen, Pierre-Auguste Renoir, Boris Kustodiev, Pyotr Konchalovsky, Gustave Dore, Edgar Degas, Camille Corot, Eugene Boudin.

- We included a section to check how accurate the model is for each painter. The results show that the model is better at recognizing artworks by some painters compared to others. For instance, it's good at recognizing paintings by Pyotr Konchalovsky (22.45%) and Theophile Steinlen (13.27%), but not as good for Pierre-Auguste Renoir (4.55%) and Boris Kustodiev (3.96%).  This suggest that the model has difficulty distinguishing the works of some painters compared to others.

#### ResNet18
In keras, ResNet18 doesn't exist, so we've implemented a function defined in a notebook in [Kaggle](https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras). We've trained the model with dropout and data augmentation. If we look the accuracy curve and losse's, we can afirm we don't obtain a better method. 

![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/resnet18_acc.png)
![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/resnet18_loss.png)

So, we decide to focus on **ResNet50**.

#### Reducing batch size
While we were implementing resnet18 to compare the results. We were also trying to compare the accuracy between 10 classes we were classifing our dataset. We have found different errors and see the accuracy for only one artist is giving difficulties.
We have seen that trying to do it with the test is not possible because we have images that are not classified and that we can't compare. So we will try to do it with the validation images. 

Although before we are going to test if we use sparse categorical crossentropy works better for our model. After doing the testing, we compare the results with the last categorical_crossentropy and we see that this had a better running. (adam50_sparse_categorical_crossentropy_10_artists - orange)

Another thing we can try is modifying the batch size and trying a smaller one, to see if this give us a better classification. And this is the result when we change it to: BATCH_SIZE = 30 and TEST_BATCH_SIZE = 35. But we conclude that it works worse than with the batch size original (BATCH_SIZE: 58 and TEST_BATCH_SIZE: 64). (categorical_crossentropy_less_batch_size - cian)

Otherwise we also tried with another even smaller: BATCH_SIZE = 15 and TEST_BATCH_SIZE = 20. Where we can see that is working better than the last one, but not as good as the original we where using. (categorical_crossentropy_small_batch_size - purple)

<p align="center">
    <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/acc_batch_sparse.png" alt="Alt text" width="600"/> <img src="https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/ouput/loss_batch_sparse.png" alt="Alt text" width="600"/> </div>


#### Stratifying our data
Another big mistake we has was that we were splitting train/val in a random way. So, to stratify by artist the two resulting dataframes we implement:
```
    train_df, valid_df = train_test_split(
            train_dataframe,
            test_size=val_split,
            stratify=train_dataframe['artist'],
            random_state=my_seed
        )
```

#### Adaptive Learning Rate
Also, we've tried to apply an adaptative learning rate, so if the validation accuracy doesn't improve, the learning rate is reduced by a factor (0.1). With that, we allow the model to make  more precise adjustments, helping to find a deeper local minimum. We've called this function ```AdaptiveLearningRateScheduler``` and when we train the model, we pass this callback.
The following graphic gives us a good model, having overfitting, but with train and validation accuracies higher. Remember, until know the validation accuracy reached not more than 0.5.

![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/acc_lr_adapt.png)

As well, the loss_val curve has a better tendency, back then it increased almost linearly. 

![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/loss_lr_adapt.png)

Now, we look the evolution of the learning rates:

![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/learning_rate_lr_adapt.png)

We start with a high learning rate: 0.01. And there are 4 times that val_accuracy is lower the previous one, so the learnig rate decreaes from 0.01 to 0.001, and 0.0001, until 1-06, which is the minimum lr.

#### Accumulate Gradient 4 steps
With the 10 selected artists we have 3880 photos, so we try to accumulate the gradient every 4 steps, meaning that the optimizer will update the parameters every 4 steps, using the mean of the acumulated gradients.

In the last experiment, by adapting the learning rate, the validation accuracy reached 0.7851. Now we've arrived 0.8224, it's not a significant improvment, but it's a sign that we're improving our model (or the optimizer).

![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/acc_grad_accum_4steps.png)

![Alt text](https://github.com/DCC-UAB/XNAPproject-grup07/blob/main/wandb/loss_grad_accum_4steps.png)



