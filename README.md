Health care 

# Health-Care-Analysis
# healthcare_thyroid
healthcare_thyroid_differentiated_analysis

# Introduction:
This dataset comprises clinicopathologic features collected over a span of 15 years, with the objective of predicting the recurrence of well-differentiated thyroid cancer. Each entry in the dataset represents an individual patient who was monitored for a minimum of 10 years. The dataset was created as part of research in the intersection of Artificial Intelligence and Medicine, without specific funding provided for its development. The dataset contains 13 features, each representing different clinicopathologic characteristics relevant to thyroid cancer recurrence prediction. Notably, there are no recommended data splits, indicating that the dataset may be used flexibly for various analyses and modeling tasks. Importantly, the dataset does not include any sensitive information, ensuring patient privacy and confidentiality. Additionally, there are no missing values in the dataset, facilitating seamless analysis and modeling processes.

# Objectives
The primary objective of this study is to develop machine learning models capable of predicting the likelihood of recurrence in patients diagnosed with well-differentiated thyroid cancer. Despite the generally low mortality rate associated with thyroid cancer, the risk of recurrence remains a significant concern. Accurately identifying an individual patient's risk of recurrence is crucial for guiding subsequent management and follow-up protocols. In pursuit of this objective, the study will undertake exploratory data analysis (EDA) and employ data visualization techniques to gain insights into the characteristics and distributions of the clinicopathologic features present in the dataset. Through EDA and visualization, the aim is to deepen our understanding of the relationships between various features and the recurrence of thyroid cancer.

The project's primary goals encompass several key steps:

1. Ensuring data quality and consistency by addressing missing values, outliers, and duplicates.
2. Standardizing data formats and rectifying any inconsistencies in column names or values to enhance data integrity.
3. Utilizing EDA techniques to analyze data distributions, trends, and interrelationships effectively.
4. Summarizing and visually representing key features using descriptive statistics, histograms, scatter plots, and heat maps.
5. Identifying correlations between variables and uncovering potential patterns or anomalies that may influence recurrence risk.
6. Employing various visualization tools and libraries, such as matplotlib, Seaborn, and Plotly, to generate both static and interactive visualizations, including plots and charts, for comprehensive data exploration and presentation.


# About Dataset
The dataset contains 13 features, each representing different clinicopathologic characteristics relevant to thyroid cancer recurrence prediction.

1. Age: Represents the age of individuals in the dataset.

2. Gender: Indicates the gender of individuals (e.g., Male or Female).

3. Smoking: Possibly an attribute related to smoking behavior. The specific values or categories would need further exploration.

4. HX Smoking : Indicates whether individuals have a history of smoking (e.g., Yes or No).

5. HX Radiotherapy : Indicates whether individuals have a history of radiotherapy treatment (e.g., Yes or No).

6. Thyroid Function: Possibly indicates the status or function of the thyroid gland.

7. Physical Examination: Describes the results of a physical examination, likely related to the thyroid.

8. Adenopathy: Indicates the presence and location of adenopathy (enlarged lymph nodes).

9. Pathology: Describes the types of thyroid cancer based on pathology examinations, including specific subtypes like "Micropapillary Papillary," "Follicular," and "Hürthle cell."

10. Focality: Indicates whether the thyroid cancer is unifocal or multifocal.

11. Risk: Represents the risk category associated with thyroid cancer.

12. T: Represents the Tumor stage of thyroid cancer, indicating the size and extent of the primary tumor.

13. N: Represents the Node stage of thyroid cancer, indicating the involvement of nearby lymph nodes.

14. M: Represents the Metastasis stage of thyroid cancer, indicating whether the cancer has spread to distant organs.

15. Stage: Represents the overall stage of thyroid cancer based on the combination of T, N, and M stages.

16. Response: Describes the response to treatment, including categories such as 'Indeterminate,' 'Excellent,' 'Structural Incomplete,' and 'Biochemical Incomplete.'

17. Recurred: Indicates whether thyroid cancer has recurred (e.g., Yes or No).
    

# 2. Methodology
## Data Collection
Gather relevant datasets from sources such as Differentiated Thyroid Cancer Recurrence.

## Data Cleaning
Identify and address missing values, outliers, and inconsistencies in the datasets to ensure data integrity and accuracy.

## Exploratory Data Analysis (EDA)
Explore the datasets to understand distributions, trends, and relationships between variables using descriptive statistics and data visualization techniques.

## Data preprocesing Pipeline
This project employs a comprehensive data preprocessing pipeline to prepare the dataset for machine learning analysis.

## Introduction to Preprocessing Methodology:
The preprocessing pipeline aims to address data inconsistencies, handle missing values, and prepare the dataset for machine learning modeling. It involves dividing the dataset into features and the target variable, performing necessary transformations, and splitting the data into training and testing sets.

The pipeline consists of the following key steps:

### Preprocessing Step:
In the preprocessing step, the dataset was initially divided into features and the target variable. The features were stored in a variable named X, while the target variable, denoted as 'Recurred', was stored in y. Additionally, a dictionary called label2id was created to map unique labels in the target variable to numeric values, facilitating compatibility with certain machine learning algorithms.

### Data Splitting:
The dataset was then split into training and testing sets using the train_test_split function from scikit-learn. This split was stratified, ensuring that the class distribution in the resulting training and testing sets mirrored that of the original dataset and also helps preventing biases and ensuring reliable model evaluation.

### Target Variable Distribution:
Finally, the distribution of the target variable was verified in both the training and testing sets to confirm the balanced representation of classes. This step ensures that the machine learning model is trained and tested on data that accurately reflects the underlying distribution of classes, thereby improving the model's generalizability and performance.

### Categorization of Predictors:
During preprocessing, features were categorized into numerical and categorical predictors to enable tailored transformation steps. Numerical predictors, containing 'int' and 'float' data types, were stored in numerical_predictor, while categorical predictors, including 'object' and 'category' data types, were stored in categorical_predictors. This approach allows for efficient preprocessing techniques such as scaling for numerical predictors and encoding for categorical predictors, ensuring data readiness for machine learning models.

#### The data preprocessing pipeline using ColumnTransformer from scikit-learn:

1. Importing Libraries: Import necessary libraries from scikit-learn for preprocessing, including ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, and OneHotEncoder.

2. Define Preprocessing Steps:
 - For numerical variables: a) Use SimpleImputer to fill missing values with the median of the column. b) Use StandardScaler to standardize the features by removing the mean and scaling to unit variance.
 - For categorical variables: a) Use SimpleImputer to fill missing values with the most frequent value in the column. b) Use OneHotEncoder to encode categorical features as one-hot vectors.
3. Combine Preprocessing Steps: Combine the preprocessing steps for both numerical and categorical variables using ColumnTransformer. This transformer applies the defined preprocessing steps to the appropriate columns in the dataset.
4. Fit and Transform Training Data: Call the fit_transform method on the preprocessor to fit the preprocessing steps to the training data (X_train) and transform it.
5. Transform Test Data: Call the transform method on the preprocessor to apply the fitted preprocessing steps to the test data (X_test). Note that only transformation is applied to the test data since the preprocessing steps were already fitted to the training dat

#### Implementation for Categorical Predictors
 - For preprocessing the categorical predictors in the dataset, a ColumnTransformer from sklearn.compose is utilized. This transformer allows for the application of specific transformations to different columns in the dataset.
 - The preprocessing pipeline involves encoding categorical variables into binary vectors using OneHotEncoder from sklearn.preprocessing. The parameter handle_unknown='ignore' is set to handle any unseen categories during the transformation process, ensuring robustness when dealing with new data.
 - Additionally, the 'passthrough' argument is used to keep the remaining columns unchanged, preserving their original values.
 - Fit and Transform Training Data: The fit_transform method was employed on the preprocessor to both fit and transform preprocessing steps to the training data.
 - Transform Test Data: Subsequently, the transform method was utilized on the preprocessor to apply previously fitted preprocessing steps to the test data. Notably, only transformation was applied to the test data, as preprocessing steps had already   been fitted to the training data.
#### Overall Impact:
The data preprocessing pipeline plays a crucial role in ensuring data quality, accurate encoding of categorical predictors, consistency and compatibility with machine learning models, laying the foundation for accurate and reliable analysis.

#### Model Training:
The initialized model is trained using the fit method on the preprocessed training data.

#### Prediction:
Predictions are made on both the training and test data using the trained model.

#### Performance Evaluation:
Balanced accuracy scores are computed for both the training and test predictions.

#### Output:
The computed balanced accuracy scores for both the training and test sets are printed.

### we can describe the model selection and training process as follows:
### Model Selection and Training:
 - We began by defining a set of candidate models for the classification task, including Support Vector Classifier (SVC), Random Forest Classifier, Extra Trees Classifier, XGBoost Classifier, LightGBM Classifier, and CatBoost Classifier. Each model was initialized with parameters optimized for handling imbalanced datasets and ensuring reproducibility.
 - The primary criterion for model selection was their performance on unseen data. To this end, we evaluated each model's balanced accuracy score on the test set.
 - Models were trained using the preprocessed training data, consisting of standardized numerical features and one-hot encoded categorical predictors.
#### Initialization of Models:
 - Each model was configured with specific parameters to enhance its performance and adaptability to the dataset characteristics. For instance, SVC models were configured with probability=True to allow for probability estimates and class_weight='balanced' to address class imbalance.
 - Similarly, Random Forest and Extra Trees Classifiers were initialized with class_weight='balanced' to handle class imbalance effectively.
#### Model Training and Evaluation:
 - The training process involved fitting each model to the preprocessed training data (X_train_prep_models) and corresponding target labels (y_train).
 - After training, the models made predictions on both the training and test datasets (X_train_prep_models and X_test_prep_models).
 - Subsequently, the balanced accuracy scores were computed for each model on both the training and test sets using the balanced_accuracy_score function from scikit-learn.
#### Result Presentation:
 - The balanced accuracy scores obtained for each model on both the training and test sets were stored in dictionaries (accuracy_train and accuracy_test), with the model names serving as keys.
 - This systematic approach allowed for the efficient training and evaluation of multiple models, enabling the selection of the best-performing model for the classification task.
### Process Monitoring
 - classification reports were generated for both the training and test sets to provide a detailed breakdown of the model's performance across different classes ('0' and '1').

 - These evaluation metrics offer insights into the model's ability to make accurate predictions and its performance on unseen data. They serve as crucial indicators of the model's effectiveness and generalization capability.


![download-6](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/db6600d8-39a3-4087-86e2-623d35cbe0b5)

The confusion matrices provide valuable insight into the performance of our predictive model for identifying thyroid cancer recurrence. 

#### Confusion Matrix Analysis

#### Test Dataset:
- True Negatives (TN): 55
- False Positives (FP): 2
- False Negatives (FN): 0
- True Positives (TP): 20

# Visualization

![download](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/da1b41c5-08e4-4c21-a74f-763b9cbc0ebb)

The graph shows the distribution of patients with thyroid conditions across various categories. The x-axis represents the different categories, while the y-axis shows the number of patients in each category.
The categories included in the graph are:
Gender, Smoking, History of radiotherapy, Adenopathy, Physical examination, Thyroid function, Pathology, Risk , T , N , M , Stage, Response  and Focality.
- For the gender category, there are more female patients than male patients.
- For the smoking category, the majority of patients do not smoke.
- For the history of radiotherapy category, the majority of patients do not have a history of radiotherapy.
- For the adenopathy category, the majority of patients do not have adenopathy.
- For the physical examination category, the majority of patients have a clinical euthyroid condition.
- For the thyroid function category, the majority of patients have subclinical hypothyroidism.
- For the pathology category, the majority of patients have multinodular goiter.
- For the focality category, the majority of patients have unifocal goiter.
- For the Risk category, the majority of patients are with LOW risk.
- For the T(Tumor) category, the majority of patients have T2 tumor.
- For the N(Node) category, the majority of patients have NO node.
- For the M(Metastatis) category, the majority of patients have MO .
- For the Stage category, the majority of patients have Stage 1.
- For the Response category, the majority of patients have Excellent response for the treatment.


Overall, the graph provides information about the distribution of patients with thyroid conditions across various categories. This information can be used to understand the prevalence of different thyroid conditions and to identify potential risk factors.

![download-1](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/f2126053-c3ea-43f2-8776-bac79e77f69c)

 - The image is a heatmap of healthcare data for thyorid cancer patients. The x-axis shows the age of the patients, and the y-axis shows the different features of the data. The color of each cell in the heatmap represents the value of the data for that patient. 
 - The heatmap can be used to identify patterns in the data. For example, it is clear that the risk of Thyroid cancer increases with age. The heatmap also shows that there is a correlation between smoking and Thyroid cancer risk.

Overall, the heatmap is a useful tool for visualizing and exploring healthcare data. It can be used to identify patterns and trends in the data, which can lead to a better understanding of diseases and their risk factors.


![download-2](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/64534f9b-65e6-4b87-9335-0a027a313ea8)

The graph shows the distribution of the Thyroid cancer has recurred. The x-axis shows the number of times the Thyroid cancer  has recurred, and the y-axis shows the number of Thyroid cancer recurred that number of times. Most of the cases have recurred 0 times.


![download-3](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/17c2424d-ad66-4a49-9fe8-cbde1f24ff45)

 - The graph is plotted according to the cat boost training model 

 - The graph shows the feature importances of  model. The x-axis shows the feature importance, and the y-axis shows the feature name.
  
 - The most important feature is Response followed by risk and physical examination .

![download-4](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/1ed7a36c-57b1-4fbb-bdfb-5a1f9ddbc191)

 - The graph is plotted according to the XGBClassifier training model 

 - The graph shows the feature importances of  model. The x-axis shows the feature importance, and the y-axis shows the feature name.
  
 - The most important feature is Response followed by risk and Stage .

![download-5](https://github.com/Deepu2304/Health-Care-Analysis/assets/86673603/835c4111-aa8c-442e-8407-2b3cffb675d5)
- The heatmap visualizes the PPS matrix, where each cell represents the predictive power score between two features.
- Higher PPS scores indicate stronger predictive relationships between features.
- The color intensity on the heatmap corresponds to the magnitude of the PPS scores, with darker shades indicating higher scores.
- Annotations within each cell display the precise PPS score, providing additional insights into feature predictability.
- This visualization aids in identifying the most influential features for predicting target variables, facilitating feature selection and model development.

# Conclusion:

The analysis of well-differentiated thyroid cancer recurrence reveals significant insights into patient demographics and the predictive capabilities of machine learning models. With a mean age of 40.87 years and a majority of patients being female, the dataset provides valuable demographic information.We employed various machine learning models, including Bagging Classifier, Decision Tree Classifier
, Extra Trees Classifier ,Random Forest Classifier, XGB Classifier ,AdaBoost Classifier, to predict the likelihood of recurrence. Each model was optimized for handling imbalanced datasets and ensuring reproducibility.Among these models, the CatBoost Classifier demonstrated superior performance, outperforming the base model and achieving the highest evaluation metric. We further evaluated this model using additional metrics. Achieving an AUC score of 1.0000 for training and 0.9959 for testing.


In this case, the accuracy of VotingClassifier is 96~97%, the use of Ensemble Voting can reduce the occurrence of overfitting, in this analysis process, most of the columns are objects, so they are all converted through LabelEncoder, do not specifically analyze the skew and Kurtosis, because not quite able to adjust the distribution of the data, the above is what I share, thank you!
