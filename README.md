[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/L30CyvB9)

# XNAP-Painting Processing 
Write here a short summary about your project. The text must include a short introduction and the targeted goals

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```
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
