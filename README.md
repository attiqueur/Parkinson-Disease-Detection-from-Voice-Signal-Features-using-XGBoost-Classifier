# Parkinson Disease Detection from Voice Signal Features using XGBoost Classifier

In this project, I developed an XGBoost Classifier model to identify Parkinson's disease using voice signal features. The model achieved an impressive accuracy of 91.53% and an ROC-AUC score of 0.9318. I performed feature selection by eliminating highly correlated, insignificant, and constant features, and standardized numerical features using StandardScaler to enhance model performance. The trained model was saved using pickle and deployed using Flask

### Summary of the steps:

- Import necessary libraries
- Loading the dataset
- Inspecting the Dataset
- Data Visualization
- Feature Selection
- Split the data into training and test sets, and standardize the features
- Training XGBoost Model
- Evaluating the Model
- Calculating ROC-AUC Score
- Confusion Matrix
- Save the Model
- Build a flask web app
