import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def half_divider():
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    col2.divider()

def new_line():
    st.markdown("<br>", unsafe_allow_html=True)

# Congratulation
def congratulation(key):
        col1, col2, col3 = st.columns([0.7,0.7,0.7])
        if col2.button("🎉 Congratulation", key=key):
                st.balloons()
                st.markdown(":green[🥳 You Have Successfully Finished This Phase.]")


# Config
st.set_page_config(layout="centered", page_title="Click ML", page_icon="👆")


# Title Page
st.markdown("<h1 style='text-align: center; '>📚 Study Time</h1>", unsafe_allow_html=True)
new_line()
st.markdown("Welcome to Study Time! This tab is designed to help you understand the key concepts of Data Preparation and Machine Learning. Please select a topic below to get started.", unsafe_allow_html=True)
new_line()

# Tabs
tab_titles = ['🗺️ Overview 󠀠 󠀠 󠀠', '🧭 EDA 󠀠 󠀠 󠀠', "‍📀‍‍‍‍ Missing Values 󠀠󠀠 󠀠 󠀠", "🔠 Categorical Features 󠀠 󠀠 󠀠", "🧬 Scaling & Transformation 󠀠 󠀠 󠀠", "💡 Feature Engineering 󠀠 󠀠 󠀠", "✂️ Splitting the Data 󠀠 󠀠 󠀠", "🧠 ML Models 󠀠 󠀠 󠀠"]
tabs = st.tabs(tab_titles)


# Overview
with tabs[0]:
    new_line()
    st.markdown("<h2 style='text-align: center; '>🗺️ Overview</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    When you are building a Machine Learning model, you need to follow a series of steps to prepare the data and build the model. The following are the key steps in the Machine Learning process:
    
    - **📦 Data Collection**: is the process of collecting the data from various sources such as CSV files, databases, APIs, etc. One of the most famous websites for datasets is [**Kaggle**](https://www.kaggle.com/). <br> <br>
    - **🧹 Data Cleaning**: is the process of cleaning the data by removing duplicates, handling missing values, handling outliers, etc. This step is very important because at most times the data is not clean and contains a lot of missing values and outliers. <br> <br>
    - **⚙️ Data Preprocessing**: is the process of transforming the data into a format that is suitable for analysis. This includes handling categorical features, handling numerical features, scaling and transformation, etc. <br> <br>
    - **💡 Feature Engineering**: is the process that manipulate with the features itselfs. It consists of multiple steps such as feature extraction, feature transformation, and feature selection. <br> <br>
    - **✂️ Splitting the Data**: is the process of splitting the data into training, validation, and testing sets. The training set is used to train the model, the validation set is used to tune the hyperparameters, and the testing set is used to evaluate the model. <br> <br>
    - **🧠 Building Machine Learning Models**: is the process of building the Machine Learning models. There are many Machine Learning models that can be used for classification and regression tasks. Some of the most famous models are Linear Regression, Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Neural Networks. <br> <br>
    - **⚖️ Evaluating Machine Learning Models**: is the process of evaluating the Machine Learning models using various metrics such as accuracy, precision, recall, F1 score, and many more for classification tasks and mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and R-squared for regression tasks. <br> <br>
    - **📐 Tuning Hyperparameters**: is the process of tuning the hyperparameters of the Machine Learning models to get the best model. There are many hyperparameters that can be tuned for each model such as the number of estimators for Random Forest, the number of neighbors for KNN, the number of layers and neurons for Neural Networks, and many more. <br> <br>
    """, unsafe_allow_html=True)
    new_line()
    

# Exploratory Data Analysis (EDA)
with tabs[1]:
    new_line()
    st.markdown("<h2 style='text-align: center; ' id='eda'>🧭 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    # half_divider()
    new_line()
    st.markdown("Exploratory Data Analysis (EDA) is the process of analyzing data sets to summarize their main characteristics, often with visual methods. EDA is used for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. It is an important step in the Data Preparation process. EDA is also the first step in the Machine Learning process. It is important to understand the data before building a model. This will help you to choose the right model and avoid errors. EDA is also used to find patterns, spot anomalies, test hypothesis and check assumptions with the help of summary statistics and graphical representations.", unsafe_allow_html=True)
    new_line()


    st.markdown("<h6 > The following are some of the key steps in EDA:", unsafe_allow_html=True)
    st.markdown("- **Data Collection:** This is the first step in EDA. Data can be collected from various sources such as CSV files, databases, APIs, etc.", unsafe_allow_html=True)
    st.markdown("- **Data Cleaning:** This is the process of cleaning the data by removing duplicates, handling missing values, handling outliers, etc.", unsafe_allow_html=True)
    st.markdown("- **Data Preprocessing:** This is the process of transforming the data into a format that is suitable for analysis. This includes handling categorical features, handling numerical features, scaling and transformation, etc.", unsafe_allow_html=True)
    st.markdown("- **Data Visualization:** This is the process of visualizing the data using various plots such as bar plots, histograms, scatter plots, etc.", unsafe_allow_html=True)
    st.markdown("- **Data Analysis:** This is the process of analyzing the data using various statistical methods such as mean, median, mode, standard deviation, etc.", unsafe_allow_html=True)
    st.markdown("- **Data Interpretation:** This is the process of interpreting the data to draw conclusions and make decisions.", unsafe_allow_html=True)
    new_line()

    # Data Collection with the code
    st.markdown("<h6> The following are some of the key questions that can be answered using EDA:", unsafe_allow_html=True)
    st.markdown("- **What is the size of the data?**", unsafe_allow_html=True)
    st.code("""df = pd.read_csv('data.csv') 
    df.shape""", language="python")

    st.markdown("- **What are the features in the data?**", unsafe_allow_html=True)
    st.code("""df.columns""", language="python")

    st.markdown("- **What are the data types of the features?**", unsafe_allow_html=True)
    st.code("""df.dtypes""", language="python")

    st.markdown("- **What are the missing values in the data?**", unsafe_allow_html=True)
    st.code("""df.isnull().sum()""", language="python")

    st.markdown("- **What are the outliers in the data?**", unsafe_allow_html=True)
    st.code("""df.describe()""", language="python")

    st.markdown("- **What are the correlations between the features?**", unsafe_allow_html=True)
    st.code("""df.corr()""", language="python")

    st.markdown("- **What are the distributions of the features?**", unsafe_allow_html=True)
    st.code("""df.hist()""", language="python")

    st.divider()

    # EDA with selected dataset
    new_line()
    st.subheader("Select a Dataset to Perform EDA on it")
    dataset = st.selectbox("Select a dataset", ["Select", "Iris", "Titanic", "Wine Quality", "Boston Housing"])
    new_line()

    if dataset == "Iris":
        # Iris Dataset
        st.markdown("The Iris dataset is a multivariate dataset introduced by Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. The dataset consists of 150 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. The dataset is often used in data mining, classification and clustering examples and to test algorithms. The dataset is available in the scikit-learn library.", unsafe_allow_html=True)
        new_line()

        # Perform EDA Process
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['target'] = df['target'].apply(lambda x: iris.target_names[x])
        df['target'] = df['target'].astype('category')

        # Read the data
        st.subheader("Read the Data")
        st.write("You can read the data using the following code:")
        st.code("""from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target'] = df['target'].apply(lambda x: iris.target_names[x])
df['target'] = df['target'].astype('category')""", language="python")
        st.write(df)

        # Data Size
        st.subheader("Data Size")
        st.write("The size of the data is:")
        st.code("""df.shape""", language="python")
        st.write(df.shape)
        st.markdown("The data has 150 rows and 5 columns.")
        new_line()

        # Data Types
        st.subheader("Data Types")
        st.write("The data types of the features are:")
        st.code("""df.dtypes""", language="python")
        st.write(df.dtypes)
        st.markdown("The data has 4 numerical features and 1 categorical feature.")
        new_line()

        # Missing Values
        st.subheader("Missing Values")
        st.write("The missing values in the data are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The data has no missing values.")
        new_line()

        # Description
        st.subheader("Description")
        st.write("The outliers in the data are:")
        st.code("""df.describe()""", language="python")
        st.write(df.describe())
        st.markdown("The `.describe()` method is used to generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.")
        new_line()

        # Check the distribution of each feature using histograms and box plots with plotly express
        st.subheader("Distribution of Features")
        st.write("The distribution of each feature is:")

        st.markdown("<h6> Sepal Length (cm) </h6>", unsafe_allow_html=True)
        st.code("""from plotly import express as px
fig = px.histogram(df, x='sepal length (cm)', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='sepal length (cm)', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Sepal Width (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='sepal width (cm)', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='sepal width (cm)', marginal='box')
        st.write(fig)

        st.markdown("<h6> Petal Length (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='petal length (cm)', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='petal length (cm)', marginal='box')
        st.write(fig)

        st.markdown("<h6> Petal Width (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='petal width (cm)', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='petal width (cm)', marginal='box')
        st.write(fig)
        new_line()

        # Visualize the relationship between pairs of features using scatter plots
        st.subheader("Relationship between Features")
        st.write("The relationship between pairs of features is:")
        st.markdown("<h6> Sepal Length (cm) vs Sepal Width (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)', color='target')
        st.write(fig)

        st.markdown("<h6> Sepal Length (cm) vs Petal Length (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='sepal length (cm)', y='petal length (cm)', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='sepal length (cm)', y='petal length (cm)', color='target')
        st.write(fig)

        st.markdown("<h6> Sepal Length (cm) vs Petal Width (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='sepal length (cm)', y='petal width (cm)', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='sepal length (cm)', y='petal width (cm)', color='target')
        st.write(fig)

        st.markdown("<h6> Sepal Width (cm) vs Petal Length (cm) </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='sepal width (cm)', y='petal length (cm)', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='sepal width (cm)', y='petal length (cm)', color='target')
        st.write(fig)

        # Use a heatmap to examine the correlation matrix and store it on fig variable
        st.subheader("Correlation Matrix")
        st.write("The correlation matrix is:")
        st.code("""fig = px.imshow(df.corr())
fig.show()""", language="python")
        fig = px.imshow(df.corr(), color_continuous_scale='Blues')
        st.write(fig)
        new_line()
        
        # The Distribution of the Target
        st.subheader("Distribution of the Target")
        st.write("The distribution of the target is:")
        st.code("""fig = px.histogram(df, x='target')
fig.show()""", language="python")
        fig = px.histogram(df, x='target')
        st.write(fig)
        st.markdown("The target is balanced. Each class has 50 samples, so we don't have any bias in the data.")
        new_line()

        # Problem Type
        st.subheader("Problem Type")
        st.write("The problem type is:")
        st.code("""df['target'].value_counts()""", language="python")
        st.write(df['target'].value_counts())
        st.markdown("The problem type is a classification problem. That is becuase the target is categorical.")
        new_line()

        # Conclusion
        st.subheader("Conclusion")
        st.write("From the EDA process, we can conclude that the data is clean and ready for the next step in the Machine Learning process.")
        st.write("The following are the key points from the EDA process:")
        st.markdown("- The data has 150 rows and 5 columns.")
        st.markdown("- The data has 4 numerical features and 1 categorical feature.")
        st.markdown("- The data has no missing values.")
        st.markdown("- The data has no outliers.")
        st.markdown("- The data has no correlations between the features.")
        st.markdown("- The data has no distributions.")
        st.markdown("- The target is balanced. Each class has 50 samples, so we don't have any bias in the data.")
        new_line()

        congratulation("eda_iris")

    elif dataset == "Titanic":
        # Titanic Dataset
        st.markdown("The Titanic dataset is a multivariate dataset that contains data about the passengers of the Titanic. The dataset consists of 891 samples of passengers of the Titanic. The dataset is often used in data mining, classification and clustering examples and to test algorithms. The dataset is available in the scikit-learn library.", unsafe_allow_html=True)
        new_line()

        # Perform EDA Process
        titanic = pd.read_csv('./data/titanic.csv')

        # Read the data
        st.subheader("Read the Data")
        st.write("You can read the data using the following code:")
        st.code("""import pandas as pd
titanic = pd.read_csv('titanic.csv')""", language="python")
        st.write(titanic)

        # Data Size
        st.subheader("Data Size")
        st.write("The size of the data is:")
        st.code("""titanic.shape""", language="python")
        st.write(titanic.shape)
        st.markdown("The data has 891 rows and 12 columns.")
        new_line()

        # Data Types
        st.subheader("Data Types")
        st.write("The data types of the features are:")
        st.code("""titanic.dtypes""", language="python")
        st.write(titanic.dtypes)
        st.markdown("The data has 5 numerical features and 7 categorical features.")
        new_line()

        # Missing Values
        st.subheader("Missing Values")
        st.write("The missing values in the data are:")
        st.code("""titanic.isnull().sum()""", language="python")
        st.write(titanic.isnull().sum())
        st.markdown("The data has missing values in the `Age`, `Cabin`, and `Embarked` features.")
        new_line()

        # Description
        st.subheader("Description")
        st.write("Basic Statistics information about the data:")
        st.code("""titanic.describe()""", language="python")
        st.write(titanic.describe())
        st.markdown("The `.describe()` method is used to generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.")
        new_line()

        # Check the distribution of each feature using histograms and box plots with plotly express
        st.subheader("Distribution of Features")
        st.write("The distribution of each feature is:")
        st.markdown("<h6> Age </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(titanic, x='Age', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(titanic, x='Age', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Fare </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(titanic, x='Fare', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(titanic, x='Fare', marginal='box')
        st.write(fig)
        new_line()

        # Visualize the relationship between pairs of features using scatter plots
        st.subheader("Relationship between Features")
        st.write("The relationship between pairs of features is:")
        st.markdown("<h6> Age vs Fare </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(titanic, x='Age', y='Fare', color='Survived')
fig.show()""", language="python")
        fig = px.scatter(titanic, x='Age', y='Fare', color='Survived')
        st.write(fig)
        new_line()

        # Use a heatmap to examine the correlation matrix and store it on fig variable
        st.subheader("Correlation Matrix")
        st.write("The correlation matrix is:")
        st.code("""fig = px.imshow(titanic.corr())
fig.show()""", language="python")
        fig = px.imshow(titanic.corr(), color_continuous_scale='Blues')
        st.write(fig)
        new_line()
        
        # The Distribution of the Target
        st.subheader("Distribution of the Target")
        st.write("The distribution of the target is:")
        st.code("""fig = px.histogram(titanic, x='Survived')
fig.show()""", language="python")
        fig = px.histogram(titanic, x='Survived')
        st.write(fig)
        st.markdown("The target is imbalanced. The number of samples in the `Survived` class is less than the number of samples in the `Not Survived` class.")
        new_line()

        # Problem Type
        st.subheader("Problem Type")
        st.write("The problem type is:")
        st.code("""titanic['Survived'].value_counts()""", language="python")
        st.write(titanic['Survived'].value_counts())
        st.markdown("The problem type is a binary classification problem. That is becuase the target is categorical and hanve only 2 possible values (Survived, Or Not Survived).")

        # Conclusion
        st.subheader("Conclusion")
        st.write("From the EDA process, we can conclude that the data is clean and ready for the next step in the Machine Learning process.")
        st.write("The following are the key points from the EDA process:")
        st.markdown("- The data has 891 rows and 12 columns.")
        st.markdown("- The data has 5 numerical features and 7 categorical features.")
        st.markdown("- The data has missing values in the `Age`, `Cabin`, and `Embarked` features.")
        st.markdown("- The data has no outliers.")
        st.markdown("- The data has no correlations between the features.")
        st.markdown("- The data has no distributions.")
        st.markdown("- The target is imbalanced. The number of samples in the `Survived` class is less than the number of samples in the `Not Survived` class.")
        new_line()

        congratulation("eda_titanic")

    elif dataset == "Wine Quality":

        # Wine Quality Dataset
        st.markdown("The Wine Quality dataset is a multivariate dataset that contains data about the wine quality. The dataset consists of 1599 samples of wine. The dataset is often used in data mining, classification and clustering examples and to test algorithms. The dataset is available in the scikit-learn library.", unsafe_allow_html=True)
        new_line()

        # Perform EDA Process
        from sklearn.datasets import load_wine

        # Read the data
        st.subheader("Read the Data")
        st.write("You can read the data using the following code:")
        st.code("""from sklearn.datasets import load_wine
import pandas as pd
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['target'] = df['target'].astype('category')""", language="python")
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        df['target'] = df['target'].astype('category')
        st.write(df)
        new_line()

        # Data Size
        st.subheader("Data Size")
        st.write("The size of the data is:")
        st.code("""df.shape""", language="python")
        st.write(df.shape)
        st.markdown("The data has 1599 rows and 14 columns.")
        new_line()

        # Data Types
        st.subheader("Data Types")
        st.write("The data types of the features are:")
        st.code("""df.dtypes""", language="python")
        st.write(df.dtypes)
        st.markdown("The data has 13 numerical features and 1 categorical feature.")
        new_line()

        # Missing Values
        st.subheader("Missing Values")
        st.write("The missing values in the data are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The data has no missing values.")
        new_line()

        # Description
        st.subheader("Description")
        st.write("Basic Statistics information about the data:")
        st.code("""df.describe()""", language="python")
        st.write(df.describe())
        st.markdown("The `.describe()` method is used to generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.")
        new_line()

        # Check the distribution of each feature using histograms and box plots with plotly express
        st.subheader("Distribution of Features")
        st.write("The distribution of each feature is:")
        st.markdown("<h6> Alcohol </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='alcohol', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='alcohol', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Malic Acid </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='malic_acid', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='malic_acid', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Ash </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='ash', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='ash', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Alcalinity of Ash </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='alcalinity_of_ash', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='alcalinity_of_ash', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Magnesium </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='magnesium', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='magnesium', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Total Phenols </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='total_phenols', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='total_phenols', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Flavanoids </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='flavanoids', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='flavanoids', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Nonflavanoid Phenols </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='nonflavanoid_phenols', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='nonflavanoid_phenols', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Proanthocyanins </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='proanthocyanins', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='proanthocyanins', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Color Intensity </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='color_intensity', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='color_intensity', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Hue </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='hue', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='hue', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> OD280/OD315 of Diluted Wines </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='od280/od315_of_diluted_wines', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='od280/od315_of_diluted_wines', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> Proline </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='proline', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='proline', marginal='box')
        st.write(fig)
        new_line()

        # Visualize the relationship between pairs of features using scatter plots
        st.subheader("Relationship between Features")
        st.write("The relationship between pairs of features is:")
        st.markdown("<h6> Alcohol vs Malic Acid </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='alcohol', y='malic_acid', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='alcohol', y='malic_acid', color='target')
        st.write(fig)
        new_line()

        st.markdown("<h6> Alcohol vs Ash </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='alcohol', y='ash', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='alcohol', y='ash', color='target')
        st.write(fig)
        new_line()

        st.markdown("<h6> Alcohol vs Alcalinity of Ash </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='alcohol', y='alcalinity_of_ash', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='alcohol', y='alcalinity_of_ash', color='target')
        st.write(fig)
        new_line()

        st.markdown("<h6> Alcohol vs Magnesium </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='alcohol', y='magnesium', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='alcohol', y='magnesium', color='target')
        st.write(fig)

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        st.write("The correlation matrix is:")
        st.code("""fig = px.imshow(df.corr())
fig.show()""", language="python")
        fig = px.imshow(df.corr(), color_continuous_scale='Blues')
        fig.update_layout(width=650, height=650)
        st.write(fig)
        new_line()

        # The Distribution of the Target
        st.subheader("Distribution of the Target")
        st.write("The distribution of the target is:")
        st.code("""fig = px.histogram(df, x='target')
fig.show()""", language="python")
        fig = px.histogram(df, x='target')
        st.write(fig)
        st.markdown("The Target is not balanced. The number of class 1 is 71, class 0 is 59, and class 2 is 48.")
        # Problem Type
        st.subheader("Problem Type")
        st.write("The problem type is:")
        st.code("""df['target'].value_counts()""", language="python")
        st.write(df['target'].value_counts())
        st.markdown("The problem type is a classification problem. That is becuase the target is categorical.")

        # Conclusion
        st.subheader("Conclusion")
        st.write("From the EDA process, we can conclude that the data is clean and ready for the next step in the Machine Learning process.")
        st.write("The following are the key points from the EDA process:")
        st.markdown("- The data has 1599 rows and 14 columns.")
        st.markdown("- The data has 13 numerical features and 1 categorical feature.")
        st.markdown("- The data has no missing values.")
        st.markdown("- The data has no outliers.")
        st.markdown("- The data has no correlations between the features.")
        new_line()

        congratulation("eda_wine")

    elif dataset == "Boston Housing":
        
        # Boston Housing Dataset
        st.markdown("The Boston Housing dataset is a multivariate dataset that contains data about the Boston Housing. The dataset consists of 506 samples of houses in Boston. The dataset is often used in data mining, classification and clustering examples and to test algorithms. The dataset is available in the scikit-learn library.", unsafe_allow_html=True)
        new_line()

        # Perform EDA Process
        from sklearn.datasets import load_boston

        # Read the data
        st.subheader("Read the Data")
        st.write("You can read the data using the following code:")
        st.code("""from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target""", language="python")
        boston = load_boston()
        df = pd.DataFrame(boston.data, columns=boston.feature_names)
        df['target'] = boston.target
        st.write(df)
        new_line()

        # Data Size
        st.subheader("Data Size")
        st.write("The size of the data is:")
        st.code("""df.shape""", language="python")
        st.write(df.shape)
        st.markdown("The data has 506 rows and 14 columns.")
        new_line()

        # Data Types
        st.subheader("Data Types")
        st.write("The data types of the features are:")
        st.code("""df.dtypes""", language="python")
        st.write(df.dtypes)
        st.markdown("The data has 13 numerical features and 1 numerical target.")
        new_line()

        # Missing Values
        st.subheader("Missing Values")
        st.write("The missing values in the data are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The data has no missing values.")
        new_line()

        # Description
        st.subheader("Description")
        st.write("Basic Statistics information about the data:")
        st.code("""df.describe()""", language="python")
        st.write(df.describe())
        st.markdown("The `.describe()` method is used to generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.")
        new_line()

        # Check the distribution of each feature using histograms and box plots with plotly express
        st.subheader("Distribution of Features")
        st.write("The distribution of each feature is:")
        st.markdown("<h6> CRIM </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='CRIM', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='CRIM', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> ZN </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='ZN', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='ZN', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> INDUS </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='INDUS', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='INDUS', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> CHAS </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='CHAS', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='CHAS', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> NOX </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='NOX', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='NOX', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> RM </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='RM', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='RM', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> AGE </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='AGE', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='AGE', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> DIS </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='DIS', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='DIS', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> RAD </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='RAD', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='RAD', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> TAX </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='TAX', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='TAX', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> PTRATIO </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='PTRATIO', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='PTRATIO', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> B </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='B', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='B', marginal='box')
        st.write(fig)
        new_line()

        st.markdown("<h6> LSTAT </h6>", unsafe_allow_html=True)
        st.code("""fig = px.histogram(df, x='LSTAT', marginal='box')
fig.show()""", language="python")
        fig = px.histogram(df, x='LSTAT', marginal='box')
        st.write(fig)
        new_line()

        # Visualize the relationship between pairs of features using scatter plots
        st.subheader("Relationship between Features")
        st.write("The relationship between pairs of features is:")
        st.markdown("<h6> CRIM vs ZN </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='CRIM', y='ZN', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='CRIM', y='ZN', color='target')
        st.write(fig)
        new_line()

        st.markdown("<h6> CRIM vs INDUS </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='CRIM', y='INDUS', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='CRIM', y='INDUS', color='target')
        st.write(fig)
        new_line()

        st.markdown("<h6> CRIM vs CHAS </h6>", unsafe_allow_html=True)
        st.code("""fig = px.scatter(df, x='CRIM', y='CHAS', color='target')
fig.show()""", language="python")
        fig = px.scatter(df, x='CRIM', y='CHAS', color='target')
        st.write(fig)
        new_line()

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        st.write("The correlation matrix is:")
        st.code("""fig = px.imshow(df.corr())
fig.show()""", language="python")
        fig = px.imshow(df.corr(), color_continuous_scale='Blues')
        fig.update_layout(width=650, height=650)
        st.write(fig)
        new_line()

        # The Distribution of the Target
        st.subheader("Distribution of the Target")
        st.write("The distribution of the target is:")
        st.code("""fig = px.histogram(df, x='target')
fig.show()""", language="python")
        fig = px.histogram(df, x='target')
        st.write(fig)
        st.markdown("The target is normally distributed.")
        
        # Problem Type
        st.subheader("Problem Type")
        st.write("The problem type is:")
        st.code("""df['target'].value_counts()""", language="python")
        st.write(df['target'].value_counts())
        st.markdown("The problem type is a regression problem. That is becuase the target is numerical.")

        # Conclusion
        st.subheader("Conclusion")
        st.write("From the EDA process, we can conclude that the data is clean and ready for the next step in the Machine Learning process.")
        st.write("The following are the key points from the EDA process:")
        st.markdown("- The data has 506 rows and 14 columns.")
        st.markdown("- The data has 13 numerical features and 1 numerical target.")
        st.markdown("- The data has no missing values.")
        st.markdown("- The data has no outliers.")
        st.markdown("- The data has no correlations between the features.")
        new_line()

        congratulation("eda_boston")


# Missing Values
with tabs[2]:

    new_line()
    st.markdown("<h2 align='center'> ‍📀‍‍‍‍ Missing Values </h1>", unsafe_allow_html=True)
    
    # What is Missing Values?
    new_line()
    st.markdown("Missing values are values that are not stored for a variable in the observation. Missing values are represented by `NaN` or `None` in the data. Missing values are common in real-world datasets. Missing values can be caused by many reasons, such as human errors, data collection errors, or data processing errors. Missing values can cause problems in the Machine Learning process. That is because most Machine Learning algorithms cannot handle missing values. So, we need to handle missing values before we can use the data in the Machine Learning process.", unsafe_allow_html=True)
    new_line()

    # Why we should handle the missing values?
    st.markdown("#### ❓ Why we should handle the missing values?")
    st.markdown("Missing values can cause problems in the Machine Learning process. That is because most Machine Learning algorithms cannot handle missing values. So, we need to handle missing values before we can use the data in the Machine Learning process.", unsafe_allow_html=True)
    new_line()

    # How to Handle Missing Values?
    st.markdown("#### 🧐 How to Handle Missing Values?")
    st.markdown("There are many ways to handle missing values. The following are the most common ways to handle missing values:")
    new_line()

    st.markdown("#### 🌎 In General")
    st.markdown("**Drop the Missing Values:** We can drop the missing values from the data. That is the easiest way to handle missing values. However, this method is not recommended. That is because we will lose some information from the data. So, we will not use this method in this tutorial.")
    st.markdown("- **Drop the rows that contain missing values**: in this case we will lose some information from the data, also, we will lose some samples from the data. If the number of missing values is small, we can use this method, but if the number of missing values is large, we will lose a lot of information from the data.")
    st.markdown("- **Drop the columns that contain missing values**: in this case we will lose some information from the data, also, we will lose some features from the data. If the number of missing values is small, there is no need to drop the column (feature), instead, we can drop the rows (samples) that contain missing values or we can fill the missing values as we will see in the next methods. <br> If the number of missing values is large, we can drop the column (feature) that contains missing values. In large I mean that the percentage of missing values is more than 50% of the total number of samples.")
    new_line()

    st.markdown("##### 🔷 For Numerical Features")

    st.markdown("- **Fill with the mean**: we can fill the missing values with the mean of the feature. This method is recommended if the feature has no outliers. Because the mean is sensitive to outliers.")
    st.latex(r''' \mu = \frac{1}{n} \sum_{i=1}^{n} x_i ''')
    new_line()

    st.markdown("- **Fill with the median**: we can fill the missing values with the median of the feature. This method is recommended if the feature has outliers. Because the median is not sensitive to outliers.")
    st.latex(r''' \tilde{x} = \begin{cases} x_{\frac{n+1}{2}} & \text{if n is odd} \\ \frac{x_{\frac{n}{2}} + x_{\frac{n}{2}+1}}{2} & \text{if n is even} \end{cases} ''')
    new_line()

    st.markdown("- **Fill with the mode**: we can fill the missing values with the mode of the feature. This method is recommended if the feature is categorical.")
    st.latex(r''' mode = \text{the most frequent value} ''')
    new_line()

    st.markdown("##### 🔶 For Categorical Features")
    st.markdown("- **Fill with the most frequent value**: we can fill the missing values with the most frequent value of the feature.")
    st.latex(r''' mode = \text{the most frequent value} ''')
    new_line()
    new_line()

    # How to Handle Missing Values in Python?
    st.markdown("#### 🐍 How to Handle Missing Values in Python?")
    st.markdown("In this section, we will learn how to handle the missing values using the previous methods in Python.")
    new_line()
    
    # Drop the rows that contain missing values
    st.markdown("- Drop the rows that contain missing values")
    st.code("""df.dropna(axis=0, inplace=True)""", language="python")
    new_line()

    # Drop the columns that contain missing values
    st.markdown("- Drop the columns that contain missing values")
    st.code("""df.dropna(axis=1, inplace=True)""", language="python")
    new_line()
    
    # Fill with mean
    st.markdown("- Fill with the Mean")
    st.code("""df[feature] = df[feature].fillna(df[feature].mean())""", language="python")
    new_line()

    # Fill with median
    st.markdown("- Fill with the Median")
    st.code("""df[feature] = df[feature].fillna(df[feature].median())""", language="python")
    new_line()

    # Fill with mode
    st.markdown("- Fill with the Mode")
    st.code("""df[feature] = df[feature].fillna(df[feature].mode()[0])""", language="python")
    new_line()

    # Fill with the most frequent value
    st.markdown("- Fill with the Most Frequent Value")
    st.code("""df[feature] = df[feature].fillna(df[feature].mode()[0])""", language="python")

    # Perform Missing Values on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Perform Filling Missing Values on it")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality", "Boston Housing"])

    if dataset == "Select":
        pass
    
    elif dataset == "Iris":
        from sklearn.datasets import load_iris
        
        df  = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has no missing values. So, we do not need to handle missing values.")
        new_line()

        congratulation("missing_iris")

    elif dataset == "Titanic":

        df = pd.read_csv("./data/titanic.csv")
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has missing values. So, we need to handle missing values.")
        st.code("""null_val_df = df.isnull().sum()
null_val_df[null_val_df>0]""", language="python")
        null_val_tit = df.isnull().sum()
        st.write(null_val_tit[null_val_tit>0])
        new_line()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 align='left'> <b>Age</b> Feature", unsafe_allow_html=True)
            new_line()
            st.write(f"No. missing values: **{df[['Age']].isnull().sum().values[0]}** ")
            st.write(df[['Age']].describe().T[['mean','50%']])
            st.write("No Outliers")
            st.markdown("The used method: :green[Mean]")

        with col2:
            st.markdown("<h5 align='left'> <b> Cabin </b> Feature", unsafe_allow_html=True)
            new_line()
            st.write(f"No. missing values: **687**")
            st.code("df[['Age']].isnull().sum().values[0] / len(df)")
            st.write(f"The Percentage of missing **{687/len(df):.2f}%**")
            st.write("The used method: :green[Drop the Column]")

        with col3:
            st.markdown("<h5 align='left'> Embarked Feature", unsafe_allow_html=True)
            new_line()
            st.write("No. missing values: **2**")
            new_line()

            st.write("Cateogrical Feature")
            new_line()
            st.write("The used method: :green[Fill with the Most Frequent Value]")

        # Fill the age feature with the mean
        st.divider()
        st.markdown("#### Filling the missing values")
        new_line()

        st.markdown("##### The `Age` Feautre with the `Mean`")
        st.code("""df['Age'] = df['Age'].fillna(df['Age'].mean())""", language="python")
        new_line()

        # Drop the Cabin feature
        st.markdown("##### The `Cabin` Feautre with the `Drop the Column`")
        st.code("""df.drop('Cabin', axis=1, inplace=True)""", language="python")
        new_line()


        # Fill the Embarked feature with the most frequent value
        st.markdown("##### The `Embarked` Feautre with the `Fill with the Most Frequent Value`")
        st.code("""df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])""", language="python")
        new_line()

        congratulation("missing_titanic")
        
    elif dataset == "Wine Quality":
        from sklearn.datasets import load_wine

        df = load_wine()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_wine().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has no missing values. So, we do not need to handle missing values.")
        new_line()

        congratulation("missing_wine")

    elif dataset == "Boston Housing":
        from sklearn.datasets import load_boston

        df = load_boston()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_boston().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Missing Values:")
        st.markdown("The Missing Values in the Dataset are:")
        st.code("""df.isnull().sum()""", language="python")
        st.write(df.isnull().sum())
        st.markdown("The Dataset has no missing values. So, we do not need to handle missing values.")
        new_line()

        congratulation("missing_booston")


# Categorical Features
with tabs[3]:

    new_line()
    st.markdown("<h2 align='center'> ‍🔠‍‍‍‍ Categorical Features </h1>", unsafe_allow_html=True)

    # What is Categorical Features?
    new_line()
    st.markdown("Categorical features are features that have a finite set of values. Categorical features are also called nominal features. Categorical features can be divided into two types: **Ordinal Features** and **Nominal Features**.", unsafe_allow_html=True)
    new_line()

    # Ordinal Features
    st.markdown("#### 🔷 Ordinal Features")
    st.markdown("Ordinal features are categorical features that have a finite set of values that have an order. For example, the `Size` feature can have the values `Small`, `Medium`, and `Large`. The values of the `Size` feature have an order. That is because `Small` < `Medium` < `Large`. Another example is the `Education` feature. The `Education` feature can have the values `High School`, `Bachelor`, `Master`, and `Ph.D`. The values of the `Education` feature have an order. That is because `High School` < `Bachelor` < `Master` < `Ph.D`.", unsafe_allow_html=True)
    new_line()

    # Nominal Features
    st.markdown("#### 🔶 Nominal Features")
    st.markdown("Nominal features are categorical features that have a finite set of values that have no order. For example, the `Gender` feature can have the values `Male` and `Female`. The values of the `Gender` feature have no order. That is because `Male` is not less than `Female` and `Female` is not less than `Male`.", unsafe_allow_html=True)
    new_line()

    # How to Handle Categorical Features?
    st.markdown("#### 🧐 How to Handle Categorical Features?")
    st.markdown("There are many ways to handle categorical features. The following are the most common ways to handle categorical features:")

    st.markdown("- One Hot Encoding")
    st.markdown("- Ordinal Encoding")
    st.markdown("- Label Encoding")
    st.markdown("- Count Frequency Encoding")
    st.markdown("By the following section we will dive into each method and see how to implement it in Python, and how it works.")
    st.divider()

    # One Hot Encoding
    st.subheader("🥇 One Hot Encoding")
    st.markdown("One Hot Encoding is a method for encoding the categorical features to numerical ones by Transforming the categorical features into binary features. One Hot Encoding is used for nominal features. One Hot Encoding is also called Dummy Variables. One Hot Encoding is the most common method for encoding categorical features. That is because One Hot Encoding does not assume any order between the values of the categorical features.", unsafe_allow_html=True)
    new_line()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before One Hot Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']) ),width=250, height=250)

    with col2:
        st.write("**After One Hot Encoding**")
        st.dataframe(pd.DataFrame(np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,0],[1,0,0]]) ),width=250, height=250)

    new_line()
    st.write("As we can see, the categorical feature `col1` is transformed into three binary features `col1_a`, `col1_b`, and `col1_c`. The values of the categorical feature `col1` are `a`, `b`, and `c`. So, the value `a` is transformed into `[1,0,0]`, the value `b` is transformed into `[0,1,0]`, and the value `c` is transformed into `[0,0,1]`.")
    # new_line()
    st.code("""from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
df['feature'] = encoder.fit_transform(df[['feature']])""", language="python")
    new_line()
    new_line()

    # Ordinal Encoding
    st.subheader("♾️ Ordinal Encoding")
    st.markdown("Ordinal Encoding is a method for encoding the categorical features to numerical ones by Transforming the categorical features into numerical features. Ordinal Encoding is used for ordinal features.")
    new_line()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Ordinal Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']) ),width=250, height=250)

    with col2:
        st.write("**After Ordinal Encoding**")
        st.dataframe(pd.DataFrame(np.array([1,2,3,2,1]) ),width=250, height=250)

    new_line()
    st.write("As we can see, the categorical feature `col1` is transformed into a numerical feature `col1`. The values of the categorical feature `col1` are `a`, `b`, and `c`. So, the value `a` is transformed into `1`, the value `b` is transformed into `2`, and the value `c` is transformed into `3`.")
    st.code("""from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
df['feature'] = encoder.fit_transform(df[['feature']])""", language="python")
    new_line()
    new_line()

    # Label Encoding
    st.subheader("🏷️ Label Encoding")
    st.markdown("Label Encoding is a method for encoding the categorical features to numerical ones by Transforming the categorical features into numerical features. Label Encoding is used for ordinal features. Label Encoding is similar to Ordinal Encoding. The difference between Label Encoding and Ordinal Encoding is that Label Encoding does not assume any order between the values of the categorical features, and label encoding is a method that is just used for one feature, if you use it for multiple features, it will give and error with Python. So, Ordinal Encoding is better than Label Encoding.")
    new_line()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Label Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']) ),width=250, height=250)

    with col2:
        st.write("**After Label Encoding**")
        st.dataframe(pd.DataFrame(np.array([1,2,3,2,1]) ),width=250, height=250)

    new_line()
    st.write("As we can see, the categorical feature `col1` is transformed into a numerical feature `col1`. The values of the categorical feature `col1` are `a`, `b`, and `c`. So, the value `a` is transformed into `1`, the value `b` is transformed into `2`, and the value `c` is transformed into `3`.")
    st.code("""from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['feature'] = encoder.fit_transform(df['feature'])""", language="python")
    new_line()
    new_line()

    # Count Frequency Encoding
    st.subheader("〰️ Count Frequency Encoding")
    st.markdown("Count Frequency Encoding is a method for encoding the categorical features to numerical ones by Transforming the categorical features into numerical features. Count Frequency Encoding is used for nominal features. Count Frequency Encoding is similar to One Hot Encoding. The difference between Count Frequency Encoding and One Hot Encoding is that Count Frequency Encoding does not transform the categorical features into binary features, instead, it transforms the categorical features into numerical features. Count Frequency Encoding is better than One Hot Encoding. That is because Count Frequency Encoding does not increase the number of features in the data. So, Count Frequency Encoding is better than One Hot Encoding.")
    new_line()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Count Frequency Encoding**")
        st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']) ),width=250, height=250)

    with col2:
        st.write("**After Count Frequency Encoding**")
        st.dataframe(pd.DataFrame(np.array([2/5, 2/5, 1/5, 2/5, 2/5]) ),width=250, height=250)

    new_line()
    st.write("As we can see, the categorical feature `col1` is transformed into a numerical feature `col1`. The values of the categorical feature `col1` are `a`, `b`, and `c`. So, the value `a` is transformed into `2/5`, the value `b` is transformed into `2/5`, and the value `c` is transformed into `1/5`.")
    st.code("""df['feature'] = df['feature'].map(df['feature'].value_counts(normalize=True))""", language="python")

    new_line()
    new_line()

    # Perform Categorical Features on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Perform Categorical Features on it")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality", "Boston Housing"], key = "categorical_data")

    if dataset == "Iris":
        from sklearn.datasets import load_iris

        df  = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Categorical Features:")
        st.markdown("The Categorical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='object').columns""", language="python")
        st.write(df.select_dtypes(include='object').columns)
        st.markdown("The Dataset has no categorical features. So, we do not need to handle categorical features.")
        new_line()

        congratulation("categorical_iris")

    if dataset == 'Titanic':

        df = pd.read_csv("./data/titanic.csv")
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Categorical Features:")
        st.markdown("The Categorical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='object').columns""", language="python")
        col1, col2, col3 = st.columns(3)
        col1.markdown("The Categorical Features")
        col1.write(df.select_dtypes(include='object').columns)
        col2.markdown("The No. of Unique Values")
        col2.write(df.select_dtypes(include='object').nunique())
        col3.markdown("The Percentage of Unique Values")
        col3.write(df.select_dtypes(include='object').nunique() / len(df))
        st.markdown("The Dataset has categorical features. So, we need to handle categorical features.")
        new_line()

        col1, col2, col3 = st.columns(3)
        with col1:
                st.markdown("<h5 align='left'> <b> Name </b> Feature", unsafe_allow_html=True)
                new_line()
                st.write(f"No. Unique Values: **{df[['Name']].nunique().values[0]}** ")
                st.write(f"Percentage: **{df[['Name']].nunique().values[0] / len(df):.2f}%** ")
                st.write("The used method: :green[Drop the Column]")

        with col2:
             st.markdown("<h5 align='left'> Sex Feature", unsafe_allow_html=True)
             new_line()
             st.write(f"No. Unique Values: 2")
             st.write(f"Percentage: **{2 / len(df):.2f}%** ")
             st.write("The used method: :green[One Hot Encoding]")

        with col3:
                st.markdown("<h5 align='left'> Embarked Feature", unsafe_allow_html=True)
                new_line()
                st.write(f"No. Unique Values: **{df[['Embarked']].nunique().values[0]}** ")
                st.write(f"Percentage: **{df[['Embarked']].nunique().values[0] / len(df):.2f}%** ")
                st.write("The used method: :green[One Hot Encoding]")

        new_line()
        new_line()
        col1, col2 = st.columns(2)

        with col1:
                st.markdown("<h5 align='left'> <b> Ticket </b> Feature", unsafe_allow_html=True)
                new_line()
                st.write(f"No. Unique Values: **{df[['Ticket']].nunique().values[0]}** ")
                st.write(f"Percentage: **{df[['Ticket']].nunique().values[0] / len(df):.2f}%** ")
                st.write("The used method: :green[Drop the Column]")

        with col2:
                st.markdown("<h5 align='left'> Cabin Feature", unsafe_allow_html=True)
                new_line()
                st.write(f"No. Unique Values: **{df[['Cabin']].nunique().values[0]}** ")
                st.write(f"Percentage: **{df[['Cabin']].nunique().values[0] / len(df):.2f}%** ")
                st.write("The used method: :green[Drop the Column]")


        st.divider()
        new_line()
        st.markdown("#### Encoding the Categorical Features")
        new_line()

        # Drop the Name feature
        st.markdown("##### The `Name` Feautre with the `Drop the Column`")
        st.code("""df.drop('Name', axis=1, inplace=True)""", language="python")
        new_line()

        # One Hot Encoding the Sex & Embarked features
        st.markdown("##### The `Sex` & `Embarked` Feautres with the `One Hot Encoding`")
        st.code("df = pd.get_dummies(df, columns=['Sex','Embarked'])", language="python")
        new_line()

        # Drop the Ticket & Cabin features
        st.markdown("##### The `Ticket` & `Cabin` Feautres with the `Drop the Column`")
        st.code("""df.drop(['Ticket','Cabin'], axis=1, inplace=True)""", language="python")
        new_line()

        congratulation("categorical_titanic")

    if dataset == "Wine Quality":
        from sklearn.datasets import load_wine

        df = load_wine()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_wine().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Categorical Features:")
        st.markdown("The Categorical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='object').columns""", language="python")
        st.write(df.select_dtypes(include='object').columns)
        st.markdown("The Dataset has no categorical features. So, we do not need to handle categorical features.")
        new_line()

        congratulation("categorical_wine")

    if dataset == "Boston Housing":
        from sklearn.datasets import load_boston

        df = load_boston()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_boston().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Categorical Features:")
        st.markdown("The Categorical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='object').columns""", language="python")
        st.write(df.select_dtypes(include='object').columns)
        st.markdown("The Dataset has no categorical features. So, we do not need to handle categorical features.")
        new_line()

        congratulation("categorical_boston")


# Scaling & Transformation
with tabs[4]:

    new_line()
    st.markdown("<h2 align='center'> 🧬 Scaling & Transformation </h1>", unsafe_allow_html=True)

    # What is Scaling & Transformation?
    new_line()
    st.markdown(" :green[Data Scaling] is a method for scaling the data to a specific range, that is becuase the data can have different ranges and when a feature has a higher range, then it will have a higher impact on the model and it will add **bias**. So, we need to scale the data to a specific range.")
    st.markdown(" :green[Data Transformation] is a method for transforming the data to a specific distribution, that is becuase the data can have different distributions and when a feature has a different distribution, then it will have a higher impact on the model and it will add **bias**. So, we need to transform the data to a specific distribution. This method applied especially when the data has outliers and have high skewness.")
    new_line()

    # Why we should scale the data?
    st.markdown("##### 📏 Why we should scale the data?")
    st.markdown("Scaling the data is important for some Machine Learning algorithms. That is because some Machine Learning algorithms are sensitive to the range of the data. For example, the K-Nearest Neighbors algorithm is sensitive to the range of the data. So, we need to scale the data before we can use the data in the K-Nearest Neighbors algorithm. Another example is the Support Vector Machine algorithm. The Support Vector Machine algorithm is sensitive to the range of the data. So, we need to scale the data before we can use the data in the Support Vector Machine algorithm.", unsafe_allow_html=True)
    new_line()

    # Why we should transform the data?
    st.markdown("##### ➰ Why we should transform the data?")
    st.markdown("Transforming the data is important for some Machine Learning algorithms. That is because some Machine Learning algorithms are sensitive to the distribution of the data. For example, the Linear Regression algorithm is sensitive to the distribution of the data. So, we need to transform the data before we can use the data in the Linear Regression algorithm. Another example is the Logistic Regression algorithm. The Logistic Regression algorithm is sensitive to the distribution of the data. So, we need to transform the data before we can use the data in the Logistic Regression algorithm.", unsafe_allow_html=True)
    # new_line()
    st.divider()

    # How to scale data
    st.subheader("🧮 Scaling Methods")
    st.markdown("There are many ways to scale the data. The following are the most common ways to scale the data:")

    st.markdown("1. Min-Max Scaling")
    st.markdown("2. Standard Scaling")
    st.markdown("3. Robust Scaling")
    st.markdown("4. Max Absolute Scaling")

    st.markdown("By the following section we will dive into each method and see how to implement it in Python, and how it works.")
    new_line()

    # Min-Max Scaling
    st.markdown("##### Min-Max Scaling")
    st.markdown("Min-Max Scaling is a method for scaling the data to a specific range. Min-Max Scaling is also called Normalization. Min-Max Scaling is the most common method for scaling the data. That is because Min-Max Scaling is simple and easy to implement. Min-Max Scaling is used for features that have a normal distribution. Min-Max Scaling is used for features that have no outliers. Min-Max Scaling is used for features that have a finite range. Min-Max Scaling is used for features that have a finite set of values. Min-Max Scaling is used for features that have a finite set of values that have an order. Min-Max Scaling is used for features that have a finite set of values that have no order. Min-Max Scaling is used for features that have a finite set of values that have no order and have no outliers. Min-Max Scaling is used for features that have a finite set of values that have no order and have outliers. The range of the scaled data is from 0 to 1.", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} ''')
    col2.latex(r''' Z \in [0, 1] ''')
    new_line()

    st.code("""from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()
    new_line()

    # Standard Scaling
    st.markdown("##### Standard Scaling")
    st.markdown("Standard Scaling is a method for scaling the data to a specific range. Standard Scaling is also called Standardization. Standard Scaling is used for features that have a normal distribution. Standard Scaling is used for features that have outliers. Standard Scaling is used for features that have a finite range. Standard Scaling is used for features that have a finite set of values. Standard Scaling is used for features that have a finite set of values that have an order. Standard Scaling is used for features that have a finite set of values that have no order. Standard Scaling is used for features that have a finite set of values that have no order and have no outliers. Standard Scaling is used for features that have a finite set of values that have no order and have outliers. The range of the scaled data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x - \mu}{\sigma} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()
    new_line()

    # Robust Scaling
    st.markdown("##### Robust Scaling")
    st.markdown("Robust Scaling is a method for scaling the data to a specific range. Robust Scaling is used for features that have a normal distribution. Robust Scaling is used for features that have outliers. Robust Scaling is used for features that have a finite range. Robust Scaling is used for features that have a finite set of values. Robust Scaling is used for features that have a finite set of values that have an order. Robust Scaling is used for features that have a finite set of values that have no order. Robust Scaling is used for features that have a finite set of values that have no order and have no outliers. Robust Scaling is used for features that have a finite set of values that have no order and have outliers. The range of the scaled data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x - median}{IQR} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()
    new_line()

    # Max Absolute Scaling
    st.markdown("##### Max Absolute Scaling")
    st.markdown("Max Absolute Scaling is a method for scaling the data to a specific range. Max Absolute Scaling is used for features that have a normal distribution. Max Absolute Scaling is used for features that have outliers. Max Absolute Scaling is used for features that have a finite range. Max Absolute Scaling is used for features that have a finite set of values. Max Absolute Scaling is used for features that have a finite set of values that have an order. Max Absolute Scaling is used for features that have a finite set of values that have no order. Max Absolute Scaling is used for features that have a finite set of values that have no order and have no outliers. Max Absolute Scaling is used for features that have a finite set of values that have no order and have outliers. The range of the scaled data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{scaled} = \frac{x}{x_{max}} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df['feature'] = scaler.fit_transform(df[['feature']])""", language="python")
    new_line()
    st.divider()
    new_line()

    # How to transform data
    st.subheader("🧬 Transformation Methods")
    st.markdown("There are many ways to transform the data. The following are the most common ways to transform the data:")
    st.markdown("1. Log Transformation")
    st.markdown("2. Square Root Transformation")
    st.markdown("3. Cube Root Transformation")
    st.markdown("4. Box-Cox Transformation")

    st.markdown("By the following section we will dive into each method and see how to implement it in Python, and how it works.")
    new_line()

    # Log Transformation
    st.markdown("##### Log Transformation")
    st.markdown("Log Transformation is a method for transforming the data to a specific distribution. Log Transformation is used for features that have a right-skewed distribution. Log Transformation is used for features that have outliers. Log Transformation is used for features that have a finite range. Log Transformation is used for features that have a finite set of values. Log Transformation is used for features that have a finite set of values that have an order. Log Transformation is used for features that have a finite set of values that have no order. Log Transformation is used for features that have a finite set of values that have no order and have no outliers. Log Transformation is used for features that have a finite set of values that have no order and have outliers. The range of the transformed data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = log(x) ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""import numpy as np
df['feature'] = np.log(df['feature'])""", language="python")
    new_line()
    new_line()

    # Square Root Transformation
    st.markdown("##### Square Root Transformation")
    st.markdown("Square Root Transformation is a method for transforming the data to a specific distribution. Square Root Transformation is used for features that have a right-skewed distribution. Square Root Transformation is used for features that have outliers. Square Root Transformation is used for features that have a finite range. Square Root Transformation is used for features that have a finite set of values. Square Root Transformation is used for features that have a finite set of values that have an order. Square Root Transformation is used for features that have a finite set of values that have no order. Square Root Transformation is used for features that have a finite set of values that have no order and have no outliers. Square Root Transformation is used for features that have a finite set of values that have no order and have outliers. The range of the transformed data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \sqrt{x} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""import numpy as np
df['feature'] = np.sqrt(df['feature'])""", language="python")
    new_line()
    new_line()

    # Cube Root Transformation
    st.markdown("##### Cube Root Transformation")
    st.markdown("Cube Root Transformation is a method for transforming the data to a specific distribution. Cube Root Transformation is used for features that have a right-skewed distribution. Cube Root Transformation is used for features that have outliers. Cube Root Transformation is used for features that have a finite range. Cube Root Transformation is used for features that have a finite set of values. Cube Root Transformation is used for features that have a finite set of values that have an order. Cube Root Transformation is used for features that have a finite set of values that have no order. Cube Root Transformation is used for features that have a finite set of values that have no order and have no outliers. Cube Root Transformation is used for features that have a finite set of values that have no order and have outliers. The range of the transformed data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \sqrt[3]{x} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""import numpy as np
df['feature'] = np.cbrt(df['feature'])""", language="python")
    new_line()
    new_line()
    

    # Box-Cox Transformation
    st.markdown("##### Box-Cox Transformation")
    st.markdown("Box-Cox Transformation is a method for transforming the data to a specific distribution. Box-Cox Transformation is used for features that have a right-skewed distribution. Box-Cox Transformation is used for features that have outliers. Box-Cox Transformation is used for features that have a finite range. Box-Cox Transformation is used for features that have a finite set of values. Box-Cox Transformation is used for features that have a finite set of values that have an order. Box-Cox Transformation is used for features that have a finite set of values that have no order. Box-Cox Transformation is used for features that have a finite set of values that have no order and have no outliers. Box-Cox Transformation is used for features that have a finite set of values that have no order and have outliers. The range of the transformed data is from -1 to 1.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.latex(r''' x_{transformed} = \frac{x^{\lambda} - 1}{\lambda} ''')
    col2.latex(r''' Z \in [-1, 1] ''')
    new_line()

    st.code("""from scipy.stats import boxcox
df['feature'] = boxcox(df['feature'])[0]""", language="python")
    new_line()
    new_line()

    # Perform Scaling & Transformation on the Dataset
    st.divider()
    st.markdown("#### Select Dataset to Perform Scaling & Transformation on it")
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality", "Boston Housing"], key = "scaling_transformation_data")

    if dataset == "Iris":
        from sklearn.datasets import load_iris

        df  = load_iris()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_iris().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Numerical Features:")
        st.markdown("The Numerical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='number').columns""", language="python")
        st.write(df.select_dtypes(include='number').columns)
        st.markdown("The ranges of the features are close to each other. So, we do not need to scale the data.")

        congratulation("scale_iris")

    if dataset == 'Titanic':
         
        df = pd.read_csv("./data/titanic.csv")
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Numerical Features:")
        st.markdown("The Numerical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='number')""", language="python")
        st.markdown("The Numerical Features")
        st.write(df.select_dtypes(include='number'))
        st.markdown("The features that have different ranges with the other features are: `Age` and `Fare`. So, we need to scale them.")

        st.write("We can use any scaling method. There is no best scaling method.")
        
        # The code for each scaling method
        st.markdown("##### The `Min Max Scaler`")
        st.code("""from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])""", language="python")
        new_line()
        
        st.markdown("##### The `Standard Scaler`")
        st.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])""", language="python")
        new_line()

        st.markdown("##### The `Robust Scaler`")
        st.code("""from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])""", language="python")
        new_line()

        st.markdown("##### The `Max Absolute Scaler`")
        st.code("""from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])""", language="python")
        new_line()

        st.markdown(":red[**Very Important Note:**] :green[**_The Machine Learning Process is an Iterative Process. Which means that you might here apply `Min Max Scaler` on the `Age` and `Fare` features and build the model, and maybe when you use the `Standard Scaler` and rebuild the model it gives you better performance. In this case you should you the `Standard Scaler` becuase it gives better perfromance. <br> the idea is when you are dealing with data for building machine learning models, you will always go forth and back phases and try different method and see its reflection on the model._**]", unsafe_allow_html=True)
        new_line()
        congratulation("scale_titanic")

    if dataset == "Wine Quality":
        from sklearn.datasets import load_wine

        df = load_wine()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_wine().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Numerical Features:")
        st.markdown("The Numerical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='number')""", language="python")
        st.markdown("The Numerical Features")
        st.write(df.select_dtypes(include='number'))

        st.write("We can see  that only the `alcohol`, `alcalinity_of_ash`, and the `magnesium` features have different ranges with the other features. So, we need to scale them.")
        st.write("We can use any scaling method. There is no best scaling method.")

        # The code for each scaling method
        st.markdown("##### The `Min Max Scaler`")
        st.code("""from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['alcohol','alcalinity_of_ash','magnesium']] = scaler.fit_transform(df[['alcohol','alcalinity_of_ash','magnesium']])""", language="python")
        new_line()

        st.markdown("##### The `Standard Scaler`")
        st.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['alcohol','alcalinity_of_ash','magnesium']] = scaler.fit_transform(df[['alcohol','alcalinity_of_ash','magnesium']])""", language="python")
        new_line()

        st.markdown("##### The `Robust Scaler`")
        st.code("""from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[['alcohol','alcalinity_of_ash','magnesium']] = scaler.fit_transform(df[['alcohol','alcalinity_of_ash','magnesium']])""", language="python")
        new_line()

        st.markdown("##### The `Max Absolute Scaler`")
        st.code("""from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['alcohol','alcalinity_of_ash','magnesium']] = scaler.fit_transform(df[['alcohol','alcalinity_of_ash','magnesium']])""", language="python")
        new_line()

        st.markdown(":red[**Very Important Note:**] :green[**_The Machine Learning Process is an Iterative Process. Which means that you might here apply `Min Max Scaler` on the `alcohol`, `alcalinity_of_ash`, and the `magnesium` features and build the model, and maybe when you use the `Standard Scaler` and rebuild the model it gives you better performance. In this case you should you the `Standard Scaler` becuase it gives better perfromance. <br> the idea is when you are dealing with data for building machine learning models, you will always go forth and back phases and try different method and see its reflection on the model._**]", unsafe_allow_html=True)
        new_line()

        congratulation("scale_wine")

    if dataset == "Boston Housing":
        from sklearn.datasets import load_boston

        df = load_boston()
        df = pd.DataFrame(df.data, columns=df.feature_names)
        df['target'] = load_boston().target
        st.markdown("#### The Dataset")
        st.write(df)

        st.markdown("#### The Numerical Features:")
        st.markdown("The Numerical Features in the Dataset are:")
        st.code("""df.select_dtypes(include='number')""", language="python")
        st.markdown("The Numerical Features")
        st.write(df.select_dtypes(include='number'))

        st.write("We can see  that only the `AGE`, `TAX`, and the `B` features have different ranges with the other features. So, we need to scale them.")
        st.write("We can use the `Standard Scaler` method because the `Min Max Scaler` make the output ranges between [0-1] and the `target` has higher ranges. But in general there is no best scaling method you can try whatever you want.")
        new_line()

        st.markdown("##### The `Standard Scaler`")
        st.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['AGE','TAX','B']] = scaler.fit_transform(df[['AGE','TAX','B']])""", language="python")
        new_line()
        
        st.markdown(":red[**Very Important Note:**] :green[**_The Machine Learning Process is an Iterative Process. Which means that you might here apply `Standard Scaler` on the `AGE`, `TAX`, and the `B` features and build the model, and maybe when you use the `Robust Scaler` and rebuild the model it gives you better performance. In this case you should you the `Robust Scaler` becuase it gives better perfromance. <br> the idea is when you are dealing with data for building machine learning models, you will always go forth and back phases and try different method and see its reflection on the model._**]", unsafe_allow_html=True)
        new_line()

        congratulation("scale_boston")


# Feature Engineering
with tabs[5]:

        new_line()
        st.markdown("<h2 align='center'> 💡 Feature Engineering </h1>", unsafe_allow_html=True)

        # What is Feature Engineering?
        new_line()
        st.markdown("Feature Engineering is the process to perform some operations on the features themselves. That is because the features themselves can have some information that can be useful for the model. So, we need to extract this information from the features and add it to the data. Feature Engineering is the most important step in the Machine Learning process. That is because Feature Engineering can increase the accuracy of the model. Feature Engineering can be divided into three types: **📈 Feature Extraction**, **🔄 Feature Transformation**, and **🎯 Feature Selection**.", unsafe_allow_html=True)
        new_line()

        # Feature Extraction
        st.markdown("#### 📈 Feature Extraction")
        st.markdown("Feature Extraction is the process to extract some information from the features themselves. That is because the features themselves can have some information that can be useful for the model. So, we need to extract this information from the features and add it to the data. Feature Extraction is the most important step in the Machine Learning process. That is because Feature Extraction can increase the accuracy of the model. Feature Extraction can be divided into two types: **📊 Numerical Feature Extraction** and **🔠 Categorical Feature Extraction**.", unsafe_allow_html=True)
        new_line()

        # Numerical Feature Extraction
        st.markdown("##### 📊 Numerical Feature Extraction")
        st.markdown("Numerical Feature Extraction is the process to extract some information from the numerical features themselves. That is because the numerical features themselves can have some information that can be useful for the model. So, we need to extract this information from the numerical features and add it to the data. Numerical Feature Extraction is the most important step in the Machine Learning process. That is because Numerical Feature Extraction can increase the accuracy of the model. Numerical Feature Extraction can be divided into two types: **📏 Scaling** and **🧬 Transformation**.", unsafe_allow_html=True)
        # Example
        st.markdown("###### Examples")
        st.markdown("- In this example, we will extract some information from the `Age` feature. The `Age` feature is a numerical feature. The `Age` feature has some information that can be useful for the model. That is because the `Age` feature has some information about the `Age` of the person. So, we need to extract this information from the `Age` feature and add it to the data. We will extract the `Age` feature into two features: `Age` and `Age Group`. The `Age` feature will be the same as the original `Age` feature. The `Age Group` feature will be the `Age` feature divided by 10. So, the `Age Group` feature will have the values `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, and `9`.", unsafe_allow_html=True)
        st.markdown("- Another Example. You have 3 features called: `walking distance`, `swimming distance`, and `driving distance`. You can extract the `walking distance`, `swimming distance`, and `driving distance` features into one feature called `total distance` and values for this feature is the summing of the 3 previous features..")
        new_line()

        # Categorical Feature Extraction
        st.markdown("##### 🔠 Categorical Feature Extraction")
        st.markdown("Categorical Feature Extraction is the process to extract some information from the categorical features themselves. That is because the categorical features themselves can have some information that can be useful for the model. So, we need to extract this information from the categorical features and add it to the data. Categorical Feature Extraction is the most important step in the Machine Learning process. That is because Categorical Feature Extraction can increase the accuracy of the model. Categorical Feature Extraction can be divided into two types: **🔷 Ordinal Feature Extraction** and **🔶 Nominal Feature Extraction**.", unsafe_allow_html=True)
        # Example
        st.markdown("###### Examples")
        st.markdown("- In this example, we will extract some information from the `Education` feature. The `Education` feature is a categorical feature. The `Education` feature has some information that can be useful for the model. That is because the `Education` feature has some information about the `Education` of the person. So, we need to extract this information from the `Education` feature and add it to the data. We will extract the `Education` feature into two features: `Education` and `Education Level`. The `Education` feature will be the same as the original `Education` feature. The `Education Level` feature will be the `Education` feature divided by 10. So, the `Education Level` feature will have the values `0`, `1`, `2`, `3`, and `4`.", unsafe_allow_html=True)
        new_line()
        st.divider()

        # Feature Transformation  
        st.markdown("#### 🔄 Feature Transformation")
        st.markdown("Feature Transformation is the process to transform the values for a specific feature through mathematicall equation that follows a specific logic of transform this feature. ", unsafe_allow_html=True)
        # Example
        st.markdown("###### Examples")
        st.markdown("- In this example, you have a feature called `song_duration_ms` and you want to transform this feature to `song_duration_min` by dividing the `song_duration_ms` by 3600. This becaues to transform ms to seconds you want to divide by 60, and from second to minutes you need to divide also by 60, so the total is 3600. ")
        st.markdown(":green[Note:] by transform a feature through mathematicall equation to follow a logic, sometimes with your transformation your data is scaled to a smaller range by defautl, just like the previous example, the `song_duration_ms` is scaled to a smaller range when we transform it to `song_duration_min`.", unsafe_allow_html=True)
        st.divider()

        # Feature Selection
        st.markdown("#### 🎯 Feature Selection")
        st.markdown("Feature Selection is the process to select the most important features for the model. This is becuase your data might have many feature and not all them are important to the model, so you select the most important ones from your opinion (In the next section we will know how to select the most important features from the model we built). If you have many features (e.g. more than 20 feautres) then you need to select the most important features to the model")
        # Example
        st.markdown("###### Examples")
        st.markdown("- Some feature you need to not select them (drop them) called **Orphan columns**. These features are not important for the model and it have very high unique values. For example, if you have a feature called `id` and it has a unique value for each row, then this feature is not important for the model and you need to drop it.")
        st.markdown("- Also, if you have a categorical feature, and this feature has 850 unique values, and the total number of rows is 1000, then this feature is not important for the model and you need to drop it because the uniqueness of the values is very high.")
        new_line()

        st.markdown("##### 🤫 Secret way for applying feature selection")
        st.markdown("You can build a model and then select the most important features from this model. This is becuase the model will select the most important features for the model. So, you can build a model and then select the most important features from this model. This is the secret way for applying feature selection.")
        st.markdown("The models that has the ability to select the most important features for the model are: `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, and `CatBoost`.")
        st.markdown("The most important features for the model are the features that have the highest `importance` value.")
        
        st.markdown("###### Example")
        st.code("""from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.feature_importances_""", language="python")
        st.markdown("The `model.feature_importances_` will give you the importance of each feature. The higher the value of the importance the more important the feature is.")


        st.divider()

        # Apply Feature Engineering on the Dataset
        st.markdown("#### Select Dataset to Apply Feature Engineering on it")
        dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality", "Boston Housing"], key = "feature_engineering_data")

        if dataset == "Iris":
                from sklearn.datasets import load_iris
        
                df  = load_iris()
                df = pd.DataFrame(df.data, columns=df.feature_names)
                df['target'] = load_iris().target
                st.markdown("#### The Dataset")
                st.write(df)
                
                st.write("We can't do any type of feature engineering on this dataset because the features are very important for the model. So, we can't drop any feature, and we can't extract any information from the features.")
                new_line()
        
                congratulation("feature_engineering_iris")

        if dataset == 'Titanic':
                        
                        df = pd.read_csv("./data/titanic.csv")
                        st.markdown("#### The Dataset")
                        st.write(df)
                        
                        st.markdown("#### Feature Extraction")
                        st.markdown("- We can extract feature from the `Name` feature. We can extract the `Title` from the `Name` feature. The `Title` feature will have the values `Mr`, `Mrs`, `Miss`, `Master`, and `Other`.")
                        st.code("""df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]""", language="python")
                        new_line()
                        st.markdown("- We can extract feature from both `SibSp` and `Parch` features. We can extract the `Family Size` from the `SibSp` and `Parch` features. The `Family Size` feature will have the values of summing the `SibSp` and `Parch` features together.")
                        st.code("""df['Family Size'] = df['SibSp'] + df['Parch']""", language="python")

                        st.markdown("#### Feature Transformation")
                        st.markdown("- We can transform the `Age` feature to `Age` feature. The new `Age` feature will have the values of the `Age` feature divided by 10.")
                        st.code("""df['Age'] = df['Age'] // 10""", language="python")
                        new_line()

                        st.markdown("#### Feature Selection")
                        st.markdown("- We can drop the `PassengerId` and the `Name` features because it is an **Orphan column**.")
                        st.code("""df.drop(['PassengerId', 'Name'], axis=1, inplace=True)""", language="python")
                        new_line()

                        congratulation("feature_engineering_titanic")

        if dataset == "Wine Quality":
             
                from sklearn.datasets import load_wine
        
                df = load_wine()
                df = pd.DataFrame(df.data, columns=df.feature_names)
                df['target'] = load_wine().target
                st.markdown("#### The Dataset")
                st.write(df)
                
                st.markdown("#### Feature Extraction")
                st.markdown("- Compute the alcohol concentration per acidity level by dividing the alcohol feature by the volatile acidity feature.")
                st.code("""df['alcohol concentration per acidity level'] = df['alcohol'] / df['volatile acidity']""", language="python")
                st.markdown("- Compute the ratio between the total sulfur dioxide and free sulfur dioxide features")
                st.code("""df['total sulfur dioxide to free sulfur dioxide ratio'] = df['total sulfur dioxide'] / df['free sulfur dioxide']""", language="python")
                new_line()

                st.markdown("#### Feature Selection")
                st.markdown("We can sellect the best features for the model. The best features for the model are: `alcohol`, `flavanoids`, `color_intensity`, `total_phenols`, `od280/od315_of_diluted_wines`, `proline`, `hue`, `malic_acid`, and the `target` absolutly.")
                st.code("""df = df[['alcohol', 'flavanoids', 'color_intensity', 'total_phenols', 'od280/od315_of_diluted_wines', 'proline', 'hue', 'malic_acid', 'target']]""", language="python")
                new_line()
                congratulation("feature_engineering_wine")

        if dataset == "Boston Housing":
                from sklearn.datasets import load_boston
        
                df = load_boston()
                df = pd.DataFrame(df.data, columns=df.feature_names)
                df['target'] = load_boston().target
                st.markdown("#### The Dataset")
                st.write(df)
                
                st.markdown("#### Feature Extraction")
                st.markdown("- Compute the ratio between the `NOX` and `INDUS` features.")
                st.code("""df['NOX to INDUS ratio'] = df['NOX'] / df['INDUS']""", language="python")
                st.markdown("- Compute the ratio between the `TAX` and `INDUS` features.")
                st.code("""df['TAX to INDUS ratio'] = df['TAX'] / df['INDUS']""", language="python")
                new_line()

                st.markdown("#### Feature Selection")
                st.markdown("We can sellect the best features for the model. The best features for the model are: `RM`, `LSTAT`, `PTRATIO`, `NOX`, `INDUS`, `TAX`, `CRIM`, `AGE`, `ZN`, `RAD`, `B`, and the `target` absolutly.")
                st.code("""df = df[['RM', 'LSTAT', 'PTRATIO', 'NOX', 'INDUS', 'TAX', 'CRIM', 'AGE', 'ZN', 'RAD', 'B', 'target']]""", language="python")
                new_line()
                congratulation("feature_engineering_boston")

# Splitting Data
with tabs[6]:
        
        new_line()
        st.markdown("<h2 align='center'> ✂️ Splitting The Data </h1>", unsafe_allow_html=True)
        new_line()

        # What is Splitting The Data?
        st.markdown("Splitting The Data is the process to split the data into three parts: **Training Data**, **Valication Data**, and **Testing Data**. Splitting the data is very important step in the Machine Learning process, this is beause you want to evaluate the model on an unseen data. So we need to split the data we have into 3 part:")
        st.markdown("1. :green[Training Data:] This data is used to train the model. It must have the highest number of rows. The percentage of the training data is 60% to 80% of the total number of rows.")
        st.markdown("2. :green[Validation Data:] This data is used for Hyperparameter Tuning. It must have the lowest number of rows. The percentage of the validation data is 10% to 20% of the total number of rows.")
        st.markdown("3. :green[Testing Data:] This data is used to final evaluation for the model. It must have the second highest number of rows. The percentage of the testing data is 20% to 30% of the total number of rows.")

        # Train and Test Split
        st.markdown("#### Train and Test Split")
        st.markdown("Sometimes you need just to split your data into training and test sets without the need for the evaluation set. In this case the split size and the code could be like the following:")
        st.table(pd.DataFrame(np.array([ ["80%", "20%"]]), columns=["Train", "Test"], index=["Split Size"]))
        st.code("""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)""", language="python")
        new_line()

        # Train, Validation, and Test Split
        st.markdown("#### Train, Validation, and Test Split")
        st.markdown("Sometimes you need to split your data into training, validation, and test sets. In this case the split size and the code could be like the following:")
        st.table(pd.DataFrame(np.array([ ["70%", "15%", "15%"]]), columns=["Train", "Validation", "Test"], index=["Split Size"]))
        st.code("""from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
        new_line()

        st.markdown("**:green[1. NOTE:]** In both cases you need to split the data into features and target. So, you need to split the data into `X` and `y`.")
        st.markdown("**:green[2. NOTE:]** when you have a very big data, you don't need to give the testing and validation sets more the 5% of the total number of rows. For Exmaple, if you have 100000 (100K) rows, then you need to give the testing and validation sets 5000 rows each (as a maximum). That is because the testing and validation sets are used for evaluation, and you don't need to evaluate the model on a very big data. So, you need to give the testing and validation sets a small number of rows.")

        # Apply Splitting The Data on the Dataset
        st.divider()
        st.markdown("#### Select Dataset to Apply Splitting The Data on it")
        dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality", "Boston Housing"], key = "splitting_data")

        if dataset == "Iris":
                from sklearn.datasets import load_iris
        
                df  = load_iris()
                df = pd.DataFrame(df.data, columns=df.feature_names)
                df['target'] = load_iris().target
                st.markdown("#### The Dataset")
                st.write(df)
                
                st.markdown("#### Dataset Shape")
                st.code("""df.shape""", language="python")
                st.write(df.shape)

                st.markdown("#### Splitting The Data")
                st.markdown("We will split the data into training and testing sets. The training set will have 80% of the total number of rows. The testing set will have 20% of the total number of rows.")
                st.code("""from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], train_size=0.8)""", language="python")
                new_line()

                congratulation("splitting_iris")

        if dataset == 'Titanic':
                                
                                df = pd.read_csv("./data/titanic.csv")
                                st.markdown("#### The Dataset")
                                st.write(df)
                                
                                st.markdown("#### Dataset Shape")
                                st.code("""df.shape""", language="python")
                                st.write(df.shape)
        
                                st.markdown("#### Splitting The Data")
                                st.markdown("We will split the data into training, validation, and testing sets. The training set will have 70% of the total number of rows. The validation set will have 15% of the total number of rows. The testing set will have 15% of the total number of rows.")
                                st.code("""from sklearn.model_selection import train_test_split
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
                                new_line()

                                congratulation("splitting_titanic")

        if dataset == "Wine Quality":
              
                from sklearn.datasets import load_wine
        
                df = load_wine()
                df = pd.DataFrame(df.data, columns=df.feature_names)
                df['target'] = load_wine().target
                st.markdown("#### The Dataset")
                st.write(df)
                
                st.markdown("#### Dataset Shape")
                st.code("""df.shape""", language="python")
                st.write(df.shape)
        
                st.markdown("#### Splitting The Data")
                st.markdown("We will split the data into training, validation, and testing sets. The training set will have 70% of the total number of rows. The validation set will have 15% of the total number of rows. The testing set will have 15% of the total number of rows.")
                st.code("""from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
                new_line()

                congratulation("splitting_wine")

        if dataset == "Boston Housing":
              
                from sklearn.datasets import load_boston
        
                df = load_boston()
                df = pd.DataFrame(df.data, columns=df.feature_names)
                df['target'] = load_boston().target
                st.markdown("#### The Dataset")
                st.write(df)
                
                st.markdown("#### Dataset Shape")
                st.code("""df.shape""", language="python")
                st.write(df.shape)
        
                st.markdown("#### Splitting The Data")
                st.markdown("We will split the data into training, validation, and testing sets. The training set will have 70% of the total number of rows. The validation set will have 15% of the total number of rows. The testing set will have 15% of the total number of rows.")
                st.code("""from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5)""", language="python")
                new_line()

                congratulation("splitting_boston")


# Building Machine Learning Models
with tabs[7]:
    
        new_line()
        st.markdown("<h2 align='center'> 🧠 Building Machine Learning Models </h1>", unsafe_allow_html=True)
        new_line()

        # What is Building Machine Learning Models?
        st.markdown("Building Machine Learning Models is the process to build a machine learning model. This is the most important step in the Machine Learning process. That is because the model is the thing that will predict the target for the new data. So, we need to build a good model that can predict the target for the new data. Building Machine Learning Models can be divided into two types: **📊 Regression Models** and **🔠 Classification Models**.", unsafe_allow_html=True)
        new_line()

        # Regression Models
        st.markdown("#### 📊 Regression Models")
        st.markdown("Regression Models is the process to build a machine learning model that can predict a numerical target. This is the most important step in the Machine Learning process. That is because the model is the thing that will predict the target for the new data. So, we need to build a good model that can predict the target for the new data. Regression Models can be divided into two types: **📈 Linear Regression** and **📉 Non-Linear Regression**.", unsafe_allow_html=True)
        new_line()

        # Linear Regression
        st.markdown("##### 📈 Linear Regression")
        st.markdown("Linear Regression is the process to build a machine learning model that can predict a numerical target. This is the most important step in the Machine Learning process. That is because the model is the thing that will predict the target for the new data. So, we need to build a good model that can predict the target for the new data. Linear Regression is the most important step in the Machine Learning process. That is because Linear Regression can increase the accuracy of the model. Linear Regression can be divided into two types: **📏 Simple Linear Regression** and **📐 Multiple Linear Regression**.", unsafe_allow_html=True)
        new_line()

        # Simple Linear Regression
        st.markdown("###### 📏 Simple Linear Regression")
        st.markdown("Simple Linear Regression is the process to build a machine learning model that can predict a numerical target. This is the most important step in the Machine Learning process. That is because the model is the thing that will predict the target for the new data. So, we need to build a good model that can predict the target for the new data. Simple Linear Regression is the most important step in the Machine Learning process. That is because Simple Linear Regression can increase the accuracy of the model. Simple Linear Regression can be divided into two types: **📏 Simple Linear Regression with One Feature** and **📏 Simple Linear Regression with Two Features**.", unsafe_allow_html=True)
        new_line()

        # Simple Linear Regression with One Feature
        st.markdown("####### 📏 Simple Linear Regression with One Feature")
        st.markdown("Simple Linear Regression with One Feature is the process to build a machine learning model that can predict a numerical target. This is the most important step in the Machine Learning process. That is because the model is the thing that will predict the target for the new data. So, we need to build a good model that can predict the target for the new data. Simple Linear Regression with One Feature is the most important step in the Machine Learning process. That is because Simple Linear Regression with One Feature can increase the accuracy of the model. Simple Linear Regression with One Feature can be divided into two types: **📏 Simple Linear Regression with One Feature and One Target** and **📏 Simple Linear Regression with One Feature and Two Targets**.", unsafe_allow_html=True)
        new_line()

        






















