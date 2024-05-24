# <p align="center">Module 21 Challenge: Deep Learning/Neural Networks with Python
## Data Analytics assignment to create a binary classifier for a nonprofit foundation that can predict whether applicants will be successful if funded by the nonprofit
### Overview
#### Purpose
The purpose of this analysis is similar in nature to the previous challenge's charge determining creditworthiness, except in this case, the nonprofit entity providing funding is looking for a way to predict whether their dollars spent will lead to successful outcomes, so that they can do the most good with the money they have available to grant to other nonprofit organizations.
#### Explanation of Data
The provided .csv file of more than 34,000 organizations that have received funding from the nonprofit through the years contains a number of columns capturing the metadata of each organization, including whether the organization utilized the money effectively. This outcome of success is what the model aims to predict, based on all other relevant metadata about each organization.
#### Methods
1. The data was preprocessed by dropping unnecessary/irrelevant organization identifying columns and combining "rare" categorical variables in a new value, "Other," for columns with more than ten unique values. Next, categorical variables were encoded using `get_dummies()` from `pandas`, and the data was split into a target array ("IS_SUCCESSFUL") and a features array (the remaining columns). These arrays were split into training and testing datasets using `train_test_split`, the datasets were scaled using `StandardScaler`, fit to the training data, then `transform`ed.
2. Using `TensorFlow`, a binary classification neural network/deep learning model was designed, compiled, trained, and evaluated. Design involved choosing the number of hidden layers and number of nodes within each layer, using an appropriate activation function for each hidden layer (`relu` or `tanh`), and creating an output layer with an appropriate activation function (`sigmoidal` in this case, since the desired result is a binary output). The model was trained with the training dataset created above, and then was evaluated for accuracy using the testing dataset.
3. Attempts to optimize the model were made with a goal of achieving 75% accuracy for the testing dataset. Optimization tactics for consideration included:
    - Dropping more columns of data;
    - Using more or fewer bins for rare occurrences in columns;
    - Increasing or decreasing the number of values for each bin;
    - Using more or fewer neurons to a hidden layer;
    - Using more or fewer hidden layers;
    - Using a different activation function in a hidden layer;
    - Using more or fewer epochs in the training regimen;
    - Using `keras-tuner` to automate the optimization process.
4. Writing a report on the model, which is included as part of this README.
### Results
#### Data Preprocessing:
- The target for the model was the "IS_SUCCESSFUL" variable, a Boolean value in the dataset.
- The features for the model were:
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT
- The "EIN" and "NAME" columns were removed from the input data because they only serve as identification; they are neither targets nor features.
#### Compiling, Training, and Evaluating the Model
- My initial model was kept as simple as possible:
    - The number of input features `len(X_train[0])` was 87.
    - One hidden layer, with the `relu` activation function, and 59 nodes.
        - `relu` was chosen because it's ideal for modeling positive, nonlinear input data for classification or regression, and is considered a good starting point according to lesson material.
        - 59 nodes were chosen as it falls between the size of the input layer (87) and the output layer (1) and was calculated according to a common heuristic to use 2/3 the size of the input layer plus the size of the output layer (58 + 1).
    - An output layer with a single unit/node, using the `sigmoid` activation function.
        - A single unit was chosen because the output is a single dependent variable.
        - `sigmoid` was chosen because it's ideal for a binary classification dataset according to lesson material.
    - 50 epochs in the training regimen, based on in-class use/experience.
- My initial model did not achieve a target performance of 75% accuracy for the testing dataset, but it did come close at **73.2%**.
    - Please see the `AlphabetSoupCharity.ipynb` file for reference.
- I made three attempts to increase the model's performance:
    1. I added a second hidden layer to the initial model, with 2/3 of the nodes of the first hidden layer + 1, then rounded up to the next highest integer; this yielded a **73.1%** accuracy on the test set.
    2. I returned to the use of a single hidden layer with 2/3 of the input layer + 1, rounded up to the next highest integer, but I preprocessed the data differently. I created ranges for the "ASK_AMT" variable, similar to the "INCOME_AMT" variable, so that this numerical data could instead be categorical, in an effort to essentially create buckets for this data, because the number of "rare" occurrences of values was understandably large in some cases. This yielded a **72.8%** accuracy for the test dataset.
    3. Lastly, I returned to the first preprocessing workflow, then turned to the use of `keras-tuner` to automate the optimization of the neural network model. Some noteable differences here for the best model included the use of three hidden layers, increasing the number of nodes from layer to layer, and initializing each with the `tanh` function. However, even this couldn't achieve 75% accuracy; it's outcome was **72.6%**.
    - Please see the `.ipynb` files contained in the `Optimization_Trials` directory for reference.
### Summary
- My final model sought to combine positive aspects from the initial and optimization attempts. I used three hidden layers, decreased nodes layer by layer using 2/3 x n + 1 as I had previously, activated each hidden layer with `tanh`, but only used 25 epochs this time based on experience gained in the initial trials, and so as to not "overfit" the model to the test data.
- This resulted in a test set accuracy of **73.0%**, still shy of the 75% goal.
- I think it would be worthwhile to look a bit more into preprocessing the data in other ways to optimize the model, especially since both my own thoughtful efforts and the `keras-tuner` automated efforts yielded roughly the same accuracy.
### Further Information
#### Prerequisites
- Familiarity with and use of the Python programming language, and [Google Colab](colab.research.google.com) to interact with .ipynb files.
#### Usage
- Download the contents of the repo (as they are) to the same directory.
- Navigate to [Google Colab](colab.research.google.com) to explore the `.ipynb` files, in this order (for clarity):
    1. AlphabetSoupCharity.ipynb
    2. AlphabetSoupCharity_Optimization_Trial_1.ipynb
    3. AlphabetSoupCharity_Optimization_Trial_2.ipynb
    4. AlphabetSoupCharity_Optimization_Trial_3.ipynb
    5. AlphabetSoupCharity_Optimization.ipynb
- If desired, clear outputs and inspect/run the code cell by cell, paying attention to prompts and comments throughout.
- Have fun exploring/playing around with the data/code! What are some ways the code could be made more clear or efficient? Are there other models/methods that could be used to improve the accuracy of the predictive model?
### License
[MIT License](https://opensource.org/licenses/MIT)
### Contact
[Email](mailto:cengelhart@gmail.com)\
[GitHub](https://github.com/cengelhart0120)
