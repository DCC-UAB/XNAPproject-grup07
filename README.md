[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/L30CyvB9)

# XNAP-Painting Processing 
In this project we are working with a set of images representing paintings and we have some information related to this artisitic works. There are many goals we can try to achieve, such as detecting the painter or the sytle, we can even predit two characteristics at the same time. 

## Code structure
We will have the data outside the repository because of the size. Otherwise, this is our relevant structure:
- our_code
    - data_augmentation_flip.ipynb, where is done the process of the data augmentation for the data
    - starting_point_modified_RMSprop.ipynb
    - starting_point_modified.ipynb
    - starting_point.ipynb
- main.py
- test.py
- train.py
- README.md, where is the presentation and explanation about the repository
- QUESTIONS.md, where we will update al least every week about the process of the tasks in order to answer our initial questions.

## Starting point
The code we've taken as our 'Starting Point' for the project focuses on artist classification based on images of their paintings. Specifically, it employs a ResNet50 neural network with transfer learning to identify one of the 38 artists who have over 200 of their paintings in the dataset being used. The code includes library imports, data preprocessing, artist selection, model setup and training, accuracy evaluation, prediction, and prediction accuracy calculation.

More specifically, after data processing and artist selection for the training set, it utilizes the ResNet50 model for artist classification and adds additional layers to tailor the model to the number of artists being used. The Adam optimizer is used, and 'categorical_crossentropy' serves as the loss function. After training the model over multiple epochs, it evaluates the model using test data with graphs analyzing accuracy and loss. Finally, it predicts artist labels for test images and calculates prediction accuracy for trained and untrained artists.


## Contributors
- Marta Monsó Cadena, marta.monso@autonoma.cat
- Mercè Li de la Torre Prats, mercelidela.torre@autonoma.cat
- Daria Andreea Buiuc Balanescu, dariaandreea.buiuc@autonoma.cat
- Ariadna Lucero Abad, ariadna.lucero@autonoma.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades, 
UAB, 2023
