import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import PIL.Image as Image

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
page_icon = Image.open("./assets/icon.png")
st.set_page_config(layout="centered", page_title="Click ML", page_icon=page_icon)

with st.sidebar:
      st.image("./assets/sb-study.png", width=200)

# Title Page
st.markdown("<h1 style='text-align: center; '>📚 StudyML</h1>", unsafe_allow_html=True)
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
    dataset = st.selectbox("Select a dataset", ["Select", "Iris", "Titanic", "Wine Quality"])
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
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"])

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
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key = "categorical_data")

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
    dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key = "scaling_transformation_data")

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
        dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key = "feature_engineering_data")

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
        dataset = st.selectbox("Select Dataset", ["Select", "Iris", "Titanic", "Wine Quality"], key = "splitting_data")

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


# Building Machine Learning Models
with tabs[7]:
    
        new_line()
        st.markdown("<h2 align='center'> 🧠 Building Machine Learning Models </h1>", unsafe_allow_html=True)
        new_line()

        

        # Introduction to Building Machine Learning Models
        st.markdown(""" Machine learning models play a crucial role in predicting outcomes and making informed decisions based on data. Building an effective machine learning model requires a systematic approach that encompasses various steps, including data preparation, model selection, training, evaluation, and deployment.

Throughout this section, we will cover **Regression Models** for predicting **Numerical Targets** (continuous values) and **Classification Models** for **Categorizing Data** (discrete data) into classes. You will learn about linear regression, decision trees, random forests, support vector machines, neural networks, and more. Additionally, we will delve into evaluating model performance, selecting optimal models, tuning hyperparameters, and deploying models in real-world scenarios.

""", unsafe_allow_html=True)

        # Tabs
        tab_titles_ml = ["  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 💫 Regression Models  󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 ", "  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 🪀 Classification Models  󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠  󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 󠀠 "]
        ml_tabs = st.tabs(tab_titles_ml)

        # Regression Models
        with ml_tabs[0]:

                st.write('\n\n\n\n\n')

                st.markdown("""
                ### 💫 Overview of Regression Models

        Regression models are a fundamental class of machine learning models used for predicting numerical values. In this section, we will explore two main types of regression models: linear regression and non-linear regression.

        #### ✏️ Linear Regression

        Linear regression is a widely used regression technique that models the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the independent variables and the dependent variable. The goal of linear regression is to find the best-fit line that minimizes the difference between the predicted and actual values.

        There are two types of linear regression:

        - **Simple Linear Regression**: In simple linear regression, we have one independent variable (feature) and one dependent variable (target). The model fits a straight line to the data to predict the target variable.

        - **Multiple Linear Regression**: Multiple linear regression extends simple linear regression to include multiple independent variables. It considers the linear relationship between multiple features and the target variable.

        #### ⛓ Non-Linear Regression

        Non-linear regression models capture more complex relationships between the independent variables and the dependent variable. Unlike linear regression, non-linear regression does not assume a linear relationship. It allows for curves, exponential functions, polynomial functions, and other non-linear patterns.

        Non-linear regression can be useful when the relationship between variables is not linear and requires a more flexible model to capture the underlying patterns in the data. Various algorithms, such as decision trees, support vector regression, and neural networks, can be used to perform non-linear regression.

        
        <br> 
                        
        ---
                        
        <br>

        """, unsafe_allow_html=True)
                
                # Regression Algorithms
                st.markdown("### 🧪 Regression Algorithms")
                st.markdown("There are numerous regression algorithms available, each with its strengths and limitations. Some popular algorithms include: **Linear Regression**, **Decision Trees**, **Random Forest**, **Support Vector Regression (SVR)**, **K Nearest Neighbors (KNN)**, and **XGBoost**. We will dive into each of these algorithms in detail in the following section.")
                st.write("\n")

                # expander 
                with st.expander("🧪 Regression Algorithms"):
                        st.write("\n")

                        # Regression Algorithms Tabs
                        tabs_reg_aglo = [" 📏 Linear Regression", " 🍁 Decision Tree", " 🌳 Random Forest", " ⛑️ Support Vector Regression", " 🏘️ K Nearest Neighbors", " 💥 XGBoost"]
                        reg_algo_tabs = st.tabs(tabs_reg_aglo)

                        # Linear Regression
                        with reg_algo_tabs[0]:
                                st.markdown("""
        <br>

        ## 📏 Linear Regression


        Linear Regression is a widely used and versatile algorithm for predicting numerical values. It models the relationship between the independent variables (features) and the dependent variable (target) by fitting a linear equation to the data. The goal is to find the best-fit line that minimizes the difference between the predicted and actual values.

        --- 

        ### Examples


        Linear Regression can be applied to various real-world scenarios, such as:

        - **Housing Prices**: Predicting house prices based on features like area, number of rooms, location, etc.
        - **Stock Market Analysis**: Forecasting stock prices based on historical data and relevant factors.
        - **Demand Forecasting**: Estimating future demand for a product based on past sales and market trends.

        --- 

        ### Plots


        Linear Regression is often visualized using scatter plots and regression lines. Scatter plots show the distribution of data points, while the regression line represents the linear relationship between the features and the target variable.

        ---

        ### Abilities


        Linear Regression offers several benefits:

        - **Interpretability**: The linear equation's coefficients provide insights into the relationship between the features and the target variable.
        - **Simplicity**: Linear Regression is easy to understand and implement, making it suitable for both beginners and experts.
        - **Efficiency**: The training and prediction process is computationally efficient, allowing for quick model development.

        ---


        ### Pros and Cons


        **Pros:**
        - Linear Regression performs well when the relationship between the features and the target is linear.
        - It provides interpretability, allowing you to understand the impact of each feature on the target.
        - Linear Regression is computationally efficient and can handle large datasets.

        **Cons:**
        - Linear Regression assumes a linear relationship, which may not be suitable for datasets with complex non-linear relationships.
        - It is sensitive to outliers, which can significantly impact the model's performance.
        - Linear Regression is limited to modeling continuous numerical variables and may not be suitable for categorical or discrete targets.

        --- 

        ### Code:

        ```python
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---

        In summary, Linear Regression is a powerful algorithm for predicting numerical values by fitting a linear equation to the data. It is widely used and offers interpretability, simplicity, and efficiency. However, it is important to consider its assumptions and limitations when applying it to real-world datasets.

        """, unsafe_allow_html=True)

                        # Decision Tree
                        with reg_algo_tabs[1]:
                                st.markdown("""

        <br>

        ### 🍁 Decision Tree

        Decision Tree is a versatile algorithm that can be used for both classification and regression tasks. In the context of regression, Decision Trees create a tree-like model of decisions and their possible consequences. It partitions the data based on the features to predict the numerical target variable.

        ---

        #### Examples

        Decision Trees can be applied to various real-world scenarios, such as:

        - **Medical Diagnosis**: Predicting a patient's blood pressure based on symptoms, age, and other medical factors.
        - **Crop Yield Prediction**: Estimating the yield of a crop based on environmental factors, soil quality, and cultivation techniques.
        - **Insurance Premium Estimation**: Determining the appropriate insurance premium for a policyholder based on risk factors like age, occupation, and health status.

        ---

        #### Plots

        Decision Trees can be visualized as tree-like structures, where each internal node represents a decision based on a feature, each branch represents an outcome, and each leaf node represents the predicted value. The depth of the tree determines the complexity of the model.


        ---

        #### Abilities

        Decision Trees offer several benefits:

        - **Interpretability**: Decision Trees provide a clear and interpretable representation of decision-making logic.
        - **Handling Non-linearity**: Decision Trees can capture non-linear relationships between features and the target variable.
        - **Feature Importance**: Decision Trees can identify important features for prediction, aiding feature selection.

        ---

        #### Pros and Cons

        **Pros:**
        - Decision Trees can handle both numerical and categorical features.
        - They can capture non-linear relationships between features and the target variable.
        - Decision Trees are computationally efficient and can handle large datasets.

        **Cons:**
        - Decision Trees are prone to overfitting, especially when the tree becomes too deep.
        - They can be sensitive to small variations in the training data, leading to different tree structures.
        - Decision Trees may not generalize well to unseen data if the training data is not representative.

        ---

        ### Code

        ```python
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        --- 

        In summary, Decision Trees are powerful regression algorithms that create tree-like models to predict numerical values. They offer interpretability, handle non-linearity, and provide feature importance information. However, caution should be exercised to prevent overfitting and ensure generalizability.


        """, unsafe_allow_html=True)
                        
                        # Random Forest
                        with reg_algo_tabs[2]:
                                st.markdown("""
        <br>

        ### 🌳 Random Forest

        Random Forest is a powerful ensemble algorithm that combines multiple Decision Trees to create a robust regression model. It belongs to the family of ensemble methods, which aim to improve prediction performance by aggregating the predictions of multiple individual models.

        ---

        #### Ensemble Algorithms

        Ensemble methods work by combining the predictions of multiple models, often referred to as base models or weak learners, to make a final prediction. These base models are trained on different subsets of the training data or with different features to introduce diversity in their predictions. The ensemble then combines these predictions using various techniques to arrive at a final prediction.

        ---

        #### Characteristics of Random Forest

        Random Forest has the following characteristics:

        - **Multiple Decision Trees**: Random Forest consists of a collection of Decision Trees, each trained on a random subset of the training data.
        - **Bootstrap Aggregation**: The subsets of data for training each Decision Tree are created through a process called bootstrap aggregation or bagging. This involves random sampling of the training data with replacement.
        - **Feature Randomness**: In addition to data randomness, Random Forest also introduces feature randomness. Each Decision Tree is trained on a random subset of features, which helps to decorrelate the trees and increase diversity.

        ---

        #### Strengths of Random Forest

        Random Forest offers several strengths as a regression algorithm:

        - **Improved Generalization**: By combining predictions from multiple Decision Trees, Random Forest reduces overfitting and improves generalization performance.
        - **Robustness to Outliers and Noisy Data**: Random Forest is less sensitive to outliers and noisy data compared to individual Decision Trees.
        - **Feature Importance**: Random Forest provides a measure of feature importance, which helps in identifying the most influential features in the prediction.

        ---

        #### Other Factors in Ensemble Methods

        Ensemble methods, including Random Forest, exhibit the following factors:

        - **Bias-Variance Tradeoff**: Ensemble methods aim to strike a balance between bias and variance. Individual models with low bias and high variance can be combined to obtain a lower overall variance while maintaining reasonable bias.
        - **Parallelizability**: Ensemble methods can be easily parallelized, allowing for efficient training and prediction on large datasets.
        - **Model Diversity**: The performance of an ensemble relies on the diversity of the individual models. The base models should be different from each other in terms of the data they are trained on or the features they use.

        ---

        ### Code

        ```python
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---

        In summary, Random Forest is an ensemble algorithm that combines multiple Decision Trees to create a robust regression model. It leverages the diversity of individual trees to improve generalization and provides feature importance information. Ensemble methods, including Random Forest, aim to strike a balance between bias and variance while benefiting from model diversity.


        """, unsafe_allow_html=True)

                        # Support Vector Regression
                        with reg_algo_tabs[3]:
                                st.markdown("""
        <br>

        ### ⛑️ Support Vector Regression

        Support Vector Regression (SVR) is a powerful regression algorithm that utilizes the principles of Support Vector Machines to perform regression tasks. Similar to its classification counterpart, SVR aims to find a hyperplane that best fits the data points while maximizing the margin.

        ---

        #### Characteristics of Support Vector Regression

        SVR has the following characteristics:

        - **Kernel Trick**: SVR employs the kernel trick, allowing it to operate in high-dimensional feature spaces without explicitly calculating the transformations. This enables SVR to capture complex nonlinear relationships between the features and the target variable.
        - **Margin Maximization**: SVR seeks to find a hyperplane that maintains a maximum margin while tolerating a certain amount of error, known as the epsilon-tube. Data points outside this epsilon-tube are considered outliers.
        - **Support Vectors**: SVR uses a subset of training data points called support vectors, which lie on or within the margin or the epsilon-tube. These support vectors heavily influence the position and orientation of the regression hyperplane.

        ---

        #### Strengths of Support Vector Regression

        Support Vector Regression offers several strengths as a regression algorithm:

        - **Flexibility**: SVR can effectively model both linear and nonlinear relationships between features and the target variable by utilizing different kernel functions, such as linear, polynomial, radial basis function (RBF), or sigmoid.
        - **Robustness to Outliers**: SVR is robust to outliers, as it focuses on maximizing the margin and is less influenced by individual data points lying outside the margin or the epsilon-tube.
        - **Regularization**: SVR incorporates a regularization parameter, C, which controls the tradeoff between minimizing the training error and the complexity of the model. This allows for controlling overfitting and improving generalization.

        ---

        #### Considerations when using Support Vector Regression

        When working with Support Vector Regression, it's important to consider the following:

        - **Feature Scaling**: Feature scaling, such as normalization or standardization, is crucial when using SVR. SVR is sensitive to the scale of the features, and unscaled features may lead to suboptimal performance.
        - **Model Complexity**: SVR's performance is highly dependent on the choice of hyperparameters, including the kernel function, regularization parameter C, and kernel-specific parameters. Proper tuning of these hyperparameters is essential for achieving optimal performance.
        - **Computational Complexity**: SVR's training time can be relatively higher compared to some other regression algorithms, especially for large datasets. Additionally, the memory requirements for storing support vectors can be significant.

        ---

        ### Code

        ```python
        from sklearn.svm import SVR
        model = SVR()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        --- 

        In summary, Support Vector Regression (SVR) is a flexible regression algorithm that can effectively model linear and nonlinear relationships. It utilizes the kernel trick, maximizes the margin, and employs support vectors to influence the regression hyperplane. SVR is robust to outliers, provides regularization, but requires proper feature scaling and hyperparameter tuning.


        """, unsafe_allow_html=True)
                        
                        # K Nearest Neighbors
                        with reg_algo_tabs[4]:
                                st.markdown("""
        <br>

        ### 🏘️ K Nearest Neighbors (KNN)

        K Nearest Neighbors (KNN) is a simple yet powerful non-parametric algorithm used for both classification and regression tasks. In the context of regression, KNN predicts the target variable of a new data point by considering the average or weighted average of its K nearest neighbors.

        --- 

        #### Characteristics of K Nearest Neighbors

        KNN exhibits the following characteristics:

        - **Lazy Learning**: KNN is often referred to as a "lazy algorithm" because it doesn't have a traditional training phase. Instead, it stores the entire training dataset and performs computations only when making predictions on new data points. This makes the training time relatively short but can lead to longer testing times.
        - **Distance-based Similarity**: KNN determines the proximity between data points based on a distance metric, commonly Euclidean distance. The K nearest neighbors of a given data point are identified based on their distance to that point.
        - **Non-Parametric**: KNN makes no assumptions about the underlying data distribution, making it a non-parametric algorithm. It doesn't require any assumptions about the functional form of the relationship between the features and the target variable.

        ---

        #### Strengths of K Nearest Neighbors

        K Nearest Neighbors offers several strengths as a regression algorithm:

        - **Flexibility**: KNN can handle both linear and non-linear relationships between features and the target variable. It is capable of capturing complex patterns in the data, making it suitable for a wide range of regression tasks.
        - **Interpretability**: KNN provides transparency in the decision-making process. Predictions are made based on the actual values of neighboring data points, allowing for easy interpretation of results.
        - **Non-Parametric Nature**: KNN's non-parametric nature makes it more robust to outliers and less sensitive to skewed data distributions compared to parametric regression algorithms.

        ---

        #### Considerations when using K Nearest Neighbors

        When working with K Nearest Neighbors, it's important to consider the following:

        - **Feature Scaling**: Feature scaling is essential when using KNN, as it relies on the distance metric to identify neighbors. Features with larger scales can dominate the distance calculation, leading to biased results. Therefore, it's recommended to scale the features before applying KNN.
        - **Choosing the Value of K**: The choice of the parameter K, representing the number of nearest neighbors, is critical. A small value of K may lead to overfitting, while a large value may lead to underfitting. It's important to experiment with different values of K and choose the optimal value through cross-validation or other evaluation techniques.
        - **Computational Complexity**: KNN's testing time can be relatively high, especially for large datasets, as it requires calculating distances between the new data point and all training data points. Therefore, efficient data structures and algorithms, such as KD-trees or Ball-trees, can be employed to speed up the nearest neighbor search process.

        ---

        ### Code

        ```python
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(k=5) # k is the number of neighbors (we will talk about it later)
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---

        In summary, K Nearest Neighbors (KNN) is a flexible and interpretable regression algorithm. Its non-parametric nature and ability to capture complex patterns make it a suitable choice for various regression tasks. However, careful consideration should be given to feature scaling, choosing the value of K, and the computational complexity associated with larger datasets.


        """, unsafe_allow_html=True)
                        
                        # XGBoost
                        with reg_algo_tabs[5]:
                                st.markdown("""
        <br>

        ### 💥 XGBoost

        XGBoost (Extreme Gradient Boosting) is a popular and powerful machine learning algorithm known for its efficiency and effectiveness in both classification and regression tasks. It is based on the concept of gradient boosting, a type of ensemble learning technique.

        ---

        #### Characteristics of XGBoost

        XGBoost exhibits the following characteristics:

        - **Ensemble Learning**: XGBoost is an ensemble learning algorithm that combines the predictions of multiple weak models, known as decision trees, to make accurate predictions. It uses a boosting technique to sequentially train new models that focus on the errors made by the previous models, thereby improving overall prediction performance.
        - **Boosting Algorithm**: XGBoost belongs to the family of boosting algorithms, where each subsequent model in the ensemble is trained to correct the mistakes of the previous models. This iterative process allows XGBoost to gradually improve the overall predictive power by combining weak learners into a strong ensemble.
        - **Gradient Optimization**: XGBoost employs gradient-based optimization techniques to find the optimal values of model parameters. It uses gradient descent algorithms to minimize a specified loss function, resulting in better model fitting and increased predictive accuracy.

        ---

        #### Strengths of XGBoost

        XGBoost offers several strengths as a regression algorithm:

        - **High Accuracy**: XGBoost is renowned for its high prediction accuracy and performance. It leverages the ensemble of decision trees and gradient optimization techniques to make accurate predictions on a wide range of regression problems.
        - **Feature Importance**: XGBoost provides insights into feature importance, allowing users to understand the relative importance of different features in the prediction process. This information can be valuable for feature selection and understanding the underlying relationships in the data.
        - **Regularization Techniques**: XGBoost offers various regularization techniques, such as L1 and L2 regularization, which can help prevent overfitting and improve the model's generalization capability.
        - **Handling Missing Data**: XGBoost has built-in capabilities to handle missing data, eliminating the need for preprocessing steps such as imputation. It can effectively handle missing values during the training and prediction phases.

        ---

        #### Considerations when using XGBoost

        When working with XGBoost, it's important to consider the following:

        - **Parameter Tuning**: XGBoost has several hyperparameters that can significantly impact its performance. It's essential to tune these parameters carefully to achieve optimal results. Techniques such as grid search and random search can be used to find the best combination of hyperparameters.
        - **Computational Complexity**: XGBoost can be computationally expensive, especially for large datasets and complex models. It's important to consider the available computational resources and training time requirements when using XGBoost.
        - **Interpretability**: As an ensemble of decision trees, the interpretability of XGBoost may be lower compared to simpler regression algorithms. However, techniques such as feature importance can provide insights into the model's behavior.

        ---

        ### Code

        ```python
        # You need to install XGBoost first using the following command:
        # pip install xgboost
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.fit(X_train, y_train)
        model.predict(X_test)
        ```

        ---
        In summary, XGBoost is a powerful regression algorithm that leverages ensemble learning and gradient optimization techniques. Its high accuracy, feature importance analysis, and built-in regularization capabilities make it a popular choice for regression tasks. Careful parameter tuning and consideration of computational resources are essential for optimal performance.


        """, unsafe_allow_html=True)               



                # Regression Evaluation Metrics
                st.markdown("""
                
<br> 
                
---
                
<br> 

## 💯 Regression Evaluation Metrics 
                
<br> 

Evaluation metrics are used to assess the performance of machine learning models. In this section, we will explore various evaluation metrics for regression models, such as mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and R-squared (R2).
The Purpose of Evaluation Metrics is to evaluate the performance of the model. We use the evaluation metrics to compare between different models and choose the best one. We also use the evaluation metrics to evaluate the model on the testing set. The evaluation metrics are different from one problem to another. For example, we use different evaluation metrics for classification problems and regression problems. In this section, we will talk about the evaluation metrics for regression problems.

<br>

""", unsafe_allow_html=True)

                # Expander
                with st.expander("💯 Regression Evaluation Metrics"):
                        
                        st.write("\n")

                        # Evaluation tabs
                        tab_titles_eval = [" 🥪 Mean Absolute Error (MAE)" , " 🌀 Mean Squared Error (MSE)", " 🌱 Root Mean Squared Error (RMSE)" , " 🎯 R-squared (R2)"]
                        eval_tabs = st.tabs(tab_titles_eval)

                        # MAE
                        with eval_tabs[0]:
                        
                                st.markdown("""
                        ## 🥪 MAE (Mean Absolute Error)

        The Mean Absolute Error (MAE) is a commonly used evaluation metric in regression tasks. It measures the average absolute difference between the predicted and actual values of a regression model. The MAE provides a simple and interpretable measure of the model's performance.

        ### Equation

        The MAE is calculated by taking the average of the absolute differences between the predicted values (y_pred) and the actual values (y_true) for a set of data points:
        """, unsafe_allow_html=True)

                                st.latex(r''' MAE = \frac{1}{n} \sum_{i=1}^{n} |y_{true} - y_{pred}| ''')
                        
                                st.markdown("""


        where:
        - $MAE$: Mean Absolute Error
        - $n$: Number of data points in the dataset
        - $Σ$: Summation symbol
        - $y_{true}$: Actual values
        - $y_{pred}$: Predicted values

        ### Usage

        MAE is used to assess the overall accuracy of a regression model. It is particularly useful when the magnitude of errors is essential and needs to be measured in the original units of the target variable. MAE provides a straightforward interpretation of the average absolute error.

        ### Interpretation

        A lower MAE value indicates better performance, as it represents a smaller average difference between the predicted and actual values. It measures the average magnitude of errors without considering their direction, making it less sensitive to outliers.

        ### Pros and Cons

        #### Pros:
        - **Intuitive Interpretation**: MAE is easy to interpret as it represents the average absolute difference between predicted and actual values.
        - **Robust to Outliers**: MAE is less affected by outliers since it treats all errors with equal importance.
        - **Same Scale as the Target Variable**: MAE is in the same units as the target variable, making it easy to relate to the problem domain.

        #### Cons:
        - **Lack of Sensitivity to Error Magnitude**: MAE treats all errors equally, regardless of their magnitude. It may not adequately penalize large errors if precise estimation of error magnitude is required.
        - **Does Not Provide Directional Information**: MAE does not indicate the direction of errors, making it difficult to identify whether the model tends to overestimate or underestimate the target variable.

        The MAE metric is a valuable tool for evaluating the performance of regression models, providing an intuitive measure of the average absolute error. However, it is essential to consider the specific requirements of the problem and the trade-offs between different evaluation metrics.

        ### Code

        ```python
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, y_pred)
        print("MAE:", mae)
        ```

                                """, unsafe_allow_html=True)        
                        
                        # MSE
                        with eval_tabs[1]:
                        
                                st.markdown("""
                        ## 🌀 MSE (Mean Squared Error)

        The Mean Squared Error (MSE) is a commonly used evaluation metric in regression tasks. It measures the average of the squared differences between the predicted and actual values of a regression model. The MSE provides a measure of the average squared error, which gives more weight to larger errors.

        ### Equation

        The MSE is calculated by taking the average of the squared differences between the predicted values (y_pred) and the actual values (y_true) for a set of data points:

        """, unsafe_allow_html= True)

                                st.latex(r''' MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2 ''')
                        
                                st.markdown("""


        where:
        - $MSE$: Mean Squared Error
        - $n$: Number of data points in the dataset
        - $Σ$: Summation symbol
        - $y_{true}$: Actual values
        - $y_{pred}$: Predicted values

        ### Usage

        MSE is used to assess the overall accuracy of a regression model. It is particularly useful when larger errors need to be penalized more compared to smaller errors. The MSE provides a measure of the average squared error.

        ### Interpretation

        A lower MSE value indicates better performance, as it represents a smaller average squared difference between the predicted and actual values. MSE measures the average magnitude of errors and penalizes larger errors more than MAE.

        ### Pros and Cons

        #### Pros:
        - **Sensitive to Large Errors**: MSE gives more weight to larger errors due to the squaring operation, making it more sensitive to outliers or extreme values.
        - **Mathematically Convenient**: Squaring the errors makes the metric mathematically convenient for optimization algorithms, as it is differentiable and enables gradient-based optimization.
        - **Same Scale as the Target Variable**: MSE is in the squared units of the target variable, which can be useful for comparing against the variance of the target variable.

        #### Cons:
        - **Lack of Intuitive Interpretation**: MSE is not as easily interpretable as MAE since it is in squared units of the target variable.
        - **Large Errors are Heavily Penalized**: MSE heavily penalizes large errors due to the squaring operation, which may not be desirable in certain applications.

        The MSE metric is commonly used to evaluate the performance of regression models, giving more weight to larger errors due to the squaring operation. However, it is important to consider the specific requirements of the problem and the trade-offs between different evaluation metrics.

        ### Code

        ```python
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        print("MSE:", mse)
        ```


        """, unsafe_allow_html=True)
                        
                        # RMSE
                        with eval_tabs[2]:
                                
                                st.markdown("""
                                ## 🌱 RMSE (Root Mean Squared Error)

        The Root Mean Squared Error (RMSE) is a popular evaluation metric in regression tasks. It is an extension of the Mean Squared Error (MSE) that addresses the issue of the MSE being in squared units. RMSE provides a measure of the average magnitude of the errors in the same unit as the target variable.

        ### Equation

        RMSE is calculated by taking the square root of the average of the squared differences between the predicted values (y_pred) and the actual values (y_true) for a set of data points:

        """, unsafe_allow_html=True)

                                st.latex(r''' RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2} ''')
                                
                                st.markdown("""

        where:

        - $RMSE$: Root Mean Squared Error
        - $n$: Number of data points in the dataset
        - $Σ$: Summation symbol
        - $y_{true}$: Actual values
        - $y_{pred}$: Predicted values
        - $\sqrt{}$: Square root

        ### Usage

        RMSE is widely used to assess the performance of regression models, especially when the error values need to be interpreted in the same unit as the target variable. It provides a measure of the average magnitude of the errors.

        ### Interpretation

        RMSE measures the square root of the average squared difference between the predicted and actual values. A lower RMSE value indicates better performance, as it represents a smaller average magnitude of the errors.

        ### Pros and Cons

        #### Pros:
        - **Interpretability**: RMSE is more easily interpretable than MSE since it is in the same unit as the target variable.
        - **Same Scale as the Target Variable**: RMSE provides a measure of the average magnitude of errors in the same unit as the target variable, which enhances interpretability.
        - **Sensitive to Large Errors**: RMSE, like MSE, gives more weight to larger errors due to the squaring operation.

        #### Cons:
        - **Lack of Intuitive Interpretation**: While RMSE is in the same unit as the target variable, its interpretation may still require domain knowledge and context.
        - **Large Errors are Heavily Penalized**: RMSE, like MSE, heavily penalizes large errors due to the squaring operation.

        The RMSE metric is widely used in regression tasks to evaluate the performance of models, providing an interpretable measure of the average magnitude of errors in the same unit as the target variable.

        ### Code

        ```python
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)
        ```

        """, unsafe_allow_html=True)

                        # R2 Score     
                        with eval_tabs[3]:
                                
                                st.markdown("""
                                ## 🎯 R-squared (R2)

        The R2 Score, also known as the coefficient of determination, is a commonly used evaluation metric for regression tasks. It measures the proportion of the variance in the dependent variable that can be explained by the independent variables.

        ### Equation

        R2 Score is calculated using the following formula:

        """, unsafe_allow_html=True)

                                st.latex(r''' R^2 = 1 - \frac{SS_{res}}{SS_{tot}} ''')
                                
                                st.markdown("""

        where:

        - $R^2$: R2 Score
        - $SS_{res}$: Sum of squared residuals
        - $SS_{tot}$: Total sum of squares

        ### Usage

        R2 Score is used to assess how well a regression model fits the given data. It provides an indication of the proportion of the variance in the dependent variable that can be explained by the independent variables.

        ### Interpretation

        The R2 Score ranges between 0 and 1. Here's how to interpret the R2 Score:

        - **R2 Score = 1**: The model perfectly predicts the target variable.
        - **R2 Score = 0**: The model fails to capture any relationship between the independent and dependent variables.
        - **R2 Score < 0**: The model performs worse than a horizontal line (the mean of the target variable).

        ### Pros and Cons

        #### Pros:
        - **Interpretability**: R2 Score provides a measure of how well the model fits the data, ranging from 0 to 1.
        - **Relative Comparison**: R2 Score allows for the comparison of different models based on their performance.
        - **Normalization**: R2 Score is normalized and does not depend on the scale of the target variable.

        #### Cons:
        - **Dependence on Model Complexity**: R2 Score may not accurately reflect the quality of the model if the model is too simple or too complex.
        - **Does Not Capture Overfitting**: R2 Score alone may not be sufficient to detect overfitting or the generalizability of the model.

        The R2 Score is a valuable metric to assess the goodness of fit of a regression model, indicating the proportion of the variance in the dependent variable that can be explained by the independent variables.

        ### Code

        ```python
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        print("R2 Score:", r2)
        ```

        """, unsafe_allow_html=True)
                                                        


        # Classification Models
        with ml_tabs[1]:
                st.write('\n\n\n\n\n')

                st.markdown("""

                ### 🪀 Overview of Classification Models

        Classification models are a class of machine learning models used for predicting categorical or discrete class labels. In this section, we will provide an overview of classification models and their key concepts.

        #### 🍴 Binary Classification

        Binary classification is a type of classification task where the goal is to predict one of two possible classes. The model learns from labeled data and assigns new instances to one of the two classes based on the learned patterns. Common algorithms used for binary classification include logistic regression, support vector machines (SVM), and decision trees.

        #### 🧊 Multiclass Classification

        Multiclass classification involves predicting one class out of three or more possible classes. It extends binary classification to handle multiple classes. The model learns from labeled data with multiple classes and assigns new instances to one of the classes. Algorithms like random forest, k-nearest neighbors (KNN), and neural networks are commonly used for multiclass classification.


        <br>

        ---

        <br>

        ### 🛹 Classification Algorithms

        There are numerous classification algorithms available, each with its strengths and limitations. Some popular algorithms include **Logistic Regression**, **Decision Trees**, **Random Forest**, **Support Vector Machines (SVM)**, **K Nearest Neighbors (KNN)**, and **XGBoost**.
        
        <br> 

         """, unsafe_allow_html=True)
                

                # expander
                with st.expander("🛹 Classification Algorithms"):
                      
                        st.write("\n")

                        # Classification Algorithms Tabs
                        tabs_class_aglo = [" 📦 Logistic Regression", " 🍁 Decision Tree", " 🌳 Random Forest", " ⛑️ Support Vector Machine", " 🏘️ K Nearest Neighbors", " 💥 XGBoost"]
                        class_algo_tabs = st.tabs(tabs_class_aglo)

                        # Logistic Regression
                        with class_algo_tabs[0]:
                                st.markdown("""

                                ## 📦 Logistic Regression

Logistic Regression is a popular algorithm for binary classification tasks. It models the relationship between the dependent variable and one or more independent variables by estimating the probabilities using a logistic function.

### How It Works

Logistic Regression works by fitting a logistic curve to the training data, which allows it to predict the probability of an instance belonging to a particular class. It uses the logistic function, also known as the sigmoid function, to map the input values to a range between 0 and 1.

### Equation

The logistic function used in Logistic Regression is given by the following equation:

""", unsafe_allow_html=True)

                                st.latex(r''' f(x) = \frac{1}{1 + e^{-x}} ''')

                                st.markdown("""

where:

- $f(x)$: Logistic function
- $e$: Euler's number
- $x$: Input value


<br> 

### Pros and Cons

#### Pros:
- **Interpretability**: Logistic Regression provides interpretable coefficients that indicate the influence of each feature on the prediction.
- **Efficiency**: Logistic Regression is computationally efficient and can handle large datasets.
- **Works well with linearly separable data**: Logistic Regression performs well when the decision boundary between classes is linear.

#### Cons:
- **Assumption of linearity**: Logistic Regression assumes a linear relationship between the independent variables and the log-odds of the dependent variable.
- **Limited to binary classification**: Logistic Regression is primarily used for binary classification tasks and may not perform well for multi-class problems without modifications.


<br>

### Code Example

Here's an example code snippet for implementing Logistic Regression using the scikit-learn library in Python:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)
           
                        # Decision Tree
                        with class_algo_tabs[1]:
                              
                                st.markdown("""

                                ## 🍁 Decision Tree

Decision Tree is a popular algorithm for classification tasks. It builds a tree-like model of decisions and their possible consequences based on the features of the data. Each internal node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome or class label.

### How It Works

Decision Tree works by recursively splitting the data based on the values of the features to create homogeneous subsets. The splits are made based on certain criteria, such as Gini impurity or information gain, to maximize the homogeneity or purity of the subsets with respect to the target variable.

<br>

### Pros and Cons

#### Pros:
- **Interpretability**: Decision Trees provide intuitive interpretations as they mimic human decision-making processes.
- **Handling both numerical and categorical data**: Decision Trees can handle both numerical and categorical features without requiring extensive preprocessing.
- **Feature importance**: Decision Trees can rank the importance of features based on their contribution to the splits.

#### Cons:
- **Overfitting**: Decision Trees have a tendency to overfit the training data, leading to poor generalization on unseen data. Techniques like pruning can be applied to alleviate this issue.
- **Instability**: Decision Trees are sensitive to small changes in the data and can produce different trees with different splits.
- **Bias towards features with more levels**: Decision Trees with categorical features tend to favor features with more levels or categories.

<br>

### Code Example

Here's an example code snippet for implementing Decision Tree using the scikit-learn library in Python:

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                        # Random Forest
                        with class_algo_tabs[2]:
                              
                                st.markdown("""

                                ## 🌳 Random Forest

Random Forest is a popular ensemble algorithm for classification tasks. It combines multiple decision trees to create a more robust and accurate model. Each decision tree in the Random Forest is built on a random subset of the data and features.

### How It Works

Random Forest works by creating an ensemble of decision trees. Each tree is trained on a random subset of the data through a process called bootstrap aggregating or "bagging." Additionally, for each split in the tree, a random subset of features is considered, reducing the correlation between trees. The final prediction is made by aggregating the predictions of all the individual trees.

### Ensemble and Boosting

Random Forest is an ensemble algorithm because it combines multiple weak learners (decision trees) to create a strong learner. Ensemble methods leverage the diversity of individual models to improve the overall prediction accuracy and reduce overfitting.

<br>

### Pros and Cons

#### Pros:
- **High Accuracy**: Random Forest tends to achieve high accuracy due to the combination of multiple decision trees.
- **Robustness**: Random Forest is resistant to overfitting and performs well on a variety of datasets.
- **Feature Importance**: Random Forest can provide feature importance scores, indicating the contribution of each feature in the classification task.

#### Cons:
- **Computational Complexity**: Random Forest can be computationally expensive, especially when dealing with large datasets or a large number of trees.
- **Lack of Interpretability**: The individual trees in the Random Forest are not easily interpretable, unlike a single decision tree.

<br>

### Code Example

Here's an example code snippet for implementing Random Forest using the scikit-learn library in Python:

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                        # Support Vector Machine
                        with class_algo_tabs[3]:
                              
                                st.markdown("""

                                ## ⛑️ Support Vector Machine

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification tasks. It finds an optimal hyperplane that best separates the different classes in the feature space.

### How It Works

SVM works by mapping the input data into a high-dimensional feature space and finding a hyperplane that maximally separates the classes. The hyperplane is determined by support vectors, which are the data points closest to the decision boundary. SVM can handle both linear and non-linear classification tasks using different kernel functions, such as linear, polynomial, and radial basis function (RBF) kernels.

### Strengths and Applications

- **Effective in High-Dimensional Spaces**: SVM performs well even in cases where the number of dimensions is greater than the number of samples. This makes it suitable for tasks with a large number of features.
- **Ability to Handle Non-Linear Data**: By using kernel functions, SVM can effectively handle non-linear classification problems by mapping the data into a higher-dimensional space.
- **Robust to Outliers**: SVM is less sensitive to outliers compared to other classification algorithms.

<br>

### Pros and Cons

#### Pros:
- **Strong Generalization**: SVM aims to find the best decision boundary with the largest margin, which often leads to good generalization performance on unseen data.
- **Effective with High-Dimensional Data**: SVM can handle high-dimensional data efficiently.
- **Flexibility with Kernels**: SVM allows the use of different kernel functions to capture complex relationships between features.

#### Cons:
- **Computationally Expensive**: SVM can be computationally expensive, especially when dealing with large datasets.
- **Sensitive to Noise**: SVM performance can be affected by noisy data, so it's important to preprocess the data and handle outliers carefully.
- **Difficult Interpretability**: SVM can be challenging to interpret, especially in high-dimensional spaces.

<br>

### Code Example

Here's an example code snippet for implementing SVM using the scikit-learn library in Python:

```python
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                        # K Nearest Neighbors
                        with class_algo_tabs[4]:
                                  
                                    st.markdown("""
        
                                    ## 🏘️ K Nearest Neighbors


K-Nearest Neighbors (KNN) is a simple yet effective supervised learning algorithm used for classification tasks. It classifies new data points based on the majority class of its k nearest neighbors in the feature space.

### How It Works

KNN works by measuring the distance between the new data point and the existing data points in the training set. It then selects the k nearest neighbors based on the chosen distance metric (e.g., Euclidean distance) and assigns the class label based on the majority vote among those neighbors.

### Strengths and Applications

- **Simplicity and Intuition**: KNN is easy to understand and implement, making it a popular choice for beginners in machine learning.
- **Non-Parametric and Lazy Learning**: KNN makes no assumptions about the underlying data distribution and does not require explicit model training. It learns from the data at the prediction stage, making it a lazy learning algorithm.
- **Ability to Handle Non-Linear Data**: KNN can effectively classify non-linear data by considering local patterns and relationships.

<br>

### Pros and Cons

#### Pros:
- **Simple Implementation**: KNN is straightforward to implement, making it an accessible algorithm for classification tasks.
- **No Training Phase**: KNN does not require an explicit training phase, as it learns from the data during prediction.
- **Non-Parametric**: KNN makes no assumptions about the underlying data distribution, giving it flexibility in handling diverse datasets.

#### Cons:
- **Computationally Expensive**: KNN can be computationally expensive, especially when dealing with large datasets or high-dimensional feature spaces.
- **Sensitive to Noise and Irrelevant Features**: KNN is sensitive to noisy data and irrelevant features, which can impact its classification accuracy.
- **Determining Optimal K**: Choosing the appropriate value of k (number of neighbors) can be challenging and may require experimentation.

<br>

### Code Example

Here's an example code snippet for implementing KNN using the scikit-learn library in Python:

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)
                        
                        # XGBoost
                        with class_algo_tabs[5]:
                                        
                                st.markdown("""
        
                                        ## 💥 XGBoost

XGBoost (Extreme Gradient Boosting) is a powerful and popular machine learning algorithm known for its exceptional performance in classification tasks. It is an ensemble learning method that combines multiple weak classifiers (decision trees) to create a strong predictive model.

### How It Works

XGBoost works by iteratively adding decision trees to improve the predictive performance. It builds trees in a sequential manner, where each subsequent tree tries to correct the errors made by the previous trees. It uses gradient boosting, a technique that minimizes a loss function by optimizing the gradient of the loss with respect to the model predictions.

### Strengths and Applications

- **High Predictive Accuracy**: XGBoost is known for its exceptional predictive accuracy and is widely used in various machine learning competitions and real-world applications.
- **Handles Complex Relationships**: XGBoost can capture complex patterns and interactions in the data, making it suitable for datasets with intricate relationships.
- **Regularization and Feature Importance**: XGBoost incorporates regularization techniques to prevent overfitting and provides feature importance scores, aiding in feature selection.

<br>

### Pros and Cons

#### Pros:
- **Highly Accurate Predictions**: XGBoost achieves state-of-the-art performance in many machine learning tasks due to its strong modeling capabilities.
- **Handles Complex Data**: XGBoost can effectively handle high-dimensional data and capture complex relationships between features.
- **Regularization and Control Overfitting**: XGBoost incorporates regularization techniques to prevent overfitting and improve generalization.

#### Cons:
- **Computationally Expensive**: XGBoost can be computationally expensive, especially when dealing with large datasets and complex models.
- **Sensitive to Hyperparameters**: XGBoost requires careful tuning of hyperparameters to achieve optimal performance, which can be a time-consuming process.
- **Requires Sufficient Data**: XGBoost typically requires a sufficient amount of data to train an accurate model and may not perform well with small datasets.

<br>

### Code Example

Here's an example code snippet for implementing XGBoost using the XGBoost library in Python:

```python
# You need to install XGBoost first using the following command:
# pip install xgboost
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

""", unsafe_allow_html=True)

                # Classification Evaluation Metrics
                st.markdown("""

<br>

---

<br>

### 💯 Evaluation Metrics


To assess the performance of classification models, various evaluation metrics are used. Some commonly used metrics include accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify instances and its overall performance. <br> <br>
The Purpose of Evaluation Metrics is to evaluate the performance of the model. We use the evaluation metrics to compare between different models and choose the best one. We also use the evaluation metrics to evaluate the model on the testing set. The evaluation metrics are different from one problem to another. For example, we use different evaluation metrics for classification problems and regression problems. In this section, we will talk about the evaluation metrics for classification problems. <br> <br> 
Some Metrics has high importance than others. For example, in the case of imbalanced data, we use the F1-score instead of accuracy. Also, in Medical problems, the recall is more important than precision.     
        """ , unsafe_allow_html=True)
                
                st.write("\n")
                # Expander
                with st.expander("💯 Classification Evaluation Metrics"):
                      
                        st.write("\n")


                        # Evaluation tabs
                        tab_titles_eval = [" 󠁪󠁪 󠁪 󠁪 󠁪 󠁪💢 Confusion Matrix 󠁪 󠁪 󠁪 󠁪 󠁪", " 󠁪 󠁪 󠁪 󠁪 󠁪🎯 Accuracy 󠁪 󠁪 󠁪 󠁪 󠁪 " , "󠁪 󠁪 󠁪 󠁪 󠁪 🌡️ Precision 󠁪 󠁪 󠁪 󠁪 󠁪", " 󠁪 󠁪 󠁪 󠁪 󠁪 📲 Recall 󠁪 󠁪 󠁪 󠁪 󠁪" , " 󠁪 󠁪 󠁪 󠁪 󠁪 ⚾ F1-score 󠁪 󠁪 󠁪 󠁪 󠁪 󠁪 "]
                        eval_tabs = st.tabs(tab_titles_eval)
                        

                        # Confusion Matrix
                        with eval_tabs[0]:
                                
                                st.markdown("""

## 󠁪󠁪󠁪󠁪󠁪💢 Confusion Matrix

The Confusion Matrix is a performance measurement for classification models that summarizes the results of the predictions made by the model on a set of test data. It provides a detailed breakdown of the model's performance by counting the true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.

<br>

### Understanding TP, TN, FP, and FN

- **True Positive (TP)**: The number of positive instances correctly predicted by the model as positive. These are the cases where the model predicted the class correctly.
- **True Negative (TN)**: The number of negative instances correctly predicted by the model as negative. These are the cases where the model predicted the absence of the class correctly.
- **False Positive (FP)**: The number of negative instances incorrectly predicted by the model as positive. These are the cases where the model predicted the presence of the class, but it was not present in reality.
- **False Negative (FN)**: The number of positive instances incorrectly predicted by the model as negative. These are the cases where the model predicted the absence of the class, but it was present in reality.

<br>

### Example Confusion Matrix

|         | Predicted Negative | Predicted Positive |
|---------|-------------------|-------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

<br>

Here's an example confusion matrix to illustrate how TP, TN, FP, and FN are arranged in a tabular form:


|         | Predicted Negative | Predicted Positive |
|---------|-------------------|-------------------|
| **Actual Negative** | 90 | 10 |
| **Actual Positive** | 15 | 85 |

<br>

In this example, we have a binary classification problem with two classes: Negative and Positive. The model correctly predicted 90 instances as Negative (TN), incorrectly predicted 10 instances as Positive (FP), incorrectly predicted 15 instances as Negative (FN), and correctly predicted 85 instances as Positive (TP).

The Confusion Matrix provides valuable insights into the performance of the classification model, allowing us to calculate various evaluation metrics such as accuracy, precision, recall, and F1-score.

<br>

### Code

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
```



""", unsafe_allow_html=True)

                        # Accuracy
                        with eval_tabs[1]:
                              
                                st.markdown("""

                                ## 󠁪󠁪󠁪󠁪󠁪🎯 Accuracy

Accuracy is a widely used evaluation metric for classification models that measures the overall correctness of the predictions made by the model. It calculates the ratio of correctly classified instances to the total number of instances in the dataset.

### Equation

Accuracy is calculated using the following equation:

""", unsafe_allow_html=True)
                                
                                st.latex(r''' Accuracy = \frac{TP + TN}{TP + TN + FP + FN} ''')

                                st.markdown("""

where:

- $TP$: True Positive
- $TN$: True Negative
- $FP$: False Positive
- $FN$: False Negative

<br>


### Usage and Interpretation

Accuracy is commonly used to assess the performance of a classification model, especially when the classes in the dataset are balanced (approximately equal number of instances for each class). It provides an overall measure of how well the model predicts both positive and negative instances.

### Pros and Cons

**Pros:**
- Provides a straightforward and intuitive evaluation of model performance.
- Suitable for balanced datasets or when the cost of misclassification for both classes is similar.
- Easy to interpret and communicate to stakeholders.

**Cons:**
- Accuracy alone may not be a reliable measure when dealing with imbalanced datasets, where one class dominates the others in terms of the number of instances.
- It does not provide information about the specific types of errors the model is making (e.g., false positives or false negatives).
- Accuracy may give misleading results when applied to datasets with varying class distributions or when the cost of misclassification differs significantly between classes.

Accuracy is a useful metric for initial assessment of a classification model's performance, but it should be complemented with other evaluation metrics, especially when dealing with imbalanced datasets or when the costs of different types of errors are not equal.

### Code

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
```

""", unsafe_allow_html=True)


                        # Precision
                        with eval_tabs[2]:
                               
                                st.markdown("""

                                ## 󠁪󠁪󠁪󠁪󠁪🌡️ Precision

Precision is an evaluation metric for classification models that measures the proportion of correctly predicted positive instances out of the total instances predicted as positive. It focuses on the accuracy of the positive predictions made by the model.

### Equation

Precision is calculated using the following equation:

""", unsafe_allow_html=True)

                                st.latex(r''' Precision = \frac{TP}{TP + FP} ''')

                                st.markdown("""

where:

- $TP$: True Positive
- $FP$: False Positive

<br>


### Usage and Interpretation

Precision is particularly useful when the goal is to minimize false positives. It provides insights into the model's ability to accurately classify positive instances and avoid false positives.

A high precision value indicates that the model has a low rate of falsely predicting positive instances, making it valuable in scenarios where false positives are costly or undesirable.

### Pros and Cons

**Pros:**
- Precision provides a specific measure of the model's accuracy in predicting positive instances.
- It focuses on minimizing false positives, making it suitable for applications where the cost of false positives is high.
- Useful for situations where the positive class is of higher importance or interest.

**Cons:**
- Precision does not take into account the instances that were incorrectly predicted as negative (false negatives).
- It may not provide a complete picture of the model's performance, especially when the goal is to minimize false negatives.
- Precision alone does not consider the true negatives and may not reflect the overall accuracy of the model.

Precision should be considered in conjunction with other evaluation metrics, such as recall or F1-score, to gain a comprehensive understanding of the model's performance in classification tasks.

### Code

```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
print("Precision:", precision)
```

""", unsafe_allow_html=True)
                                
                        # Recall
                        with eval_tabs[3]:

                                st.markdown("""
        
                                        ## 󠁪󠁪󠁪󠁪󠁪📲 Recall

Recall is an evaluation metric for classification models that measures the proportion of correctly predicted positive instances out of the total actual positive instances. It focuses on the model's ability to correctly identify positive instances.

### Equation

Recall is calculated using the following equation:

""", unsafe_allow_html=True)

                                st.latex(r''' Recall = \frac{TP}{TP + FN} ''')

                                st.markdown("""

where:

- $TP$: True Positive
- $FN$: False Negative

<br>


### Usage and Interpretation

Recall is particularly useful when the goal is to minimize false negatives. It provides insights into the model's ability to capture positive instances and avoid false negatives.

A high recall value indicates that the model has a low rate of falsely predicting negative instances, making it valuable in scenarios where false negatives are costly or undesirable.

### Pros and Cons

**Pros:**
- Recall provides a specific measure of the model's ability to capture positive instances.
- It focuses on minimizing false negatives, making it suitable for applications where the cost of false negatives is high.
- Useful for situations where the positive class is of higher importance or interest.

**Cons:**
- Recall does not take into account the instances that were incorrectly predicted as positive (false positives).
- It may not provide a complete picture of the model's performance, especially when the goal is to minimize false positives.
- Recall alone does not consider the true negatives and may not reflect the overall accuracy of the model.

Recall should be considered in conjunction with other evaluation metrics, such as precision or F1-score, to gain a comprehensive understanding of the model's performance in classification tasks.

### Code

```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
print("Recall:", recall)
```

""", unsafe_allow_html=True)

                        # F1 Score
                        with eval_tabs[4]:
                               
                                st.markdown("""

                                ## 󠁪󠁪󠁪󠁪󠁪⚾ F1-score

The F1-score is an evaluation metric for classification models that combines both precision and recall into a single measure. It provides a balance between precision and recall, making it a useful metric when both false positives and false negatives need to be minimized.

### Equation

The F1-score is calculated using the following equation:

""", unsafe_allow_html=True)
                                
                                st.latex(r''' F1-score = 2 \times \frac{Precision \times Recall}{Precision + Recall} ''')

                                st.markdown("""

where:

- $Precision$: Precision
- $Recall$: Recall

<br>


### Usage and Interpretation

The F1-score is particularly useful when there is an uneven class distribution or when false positives and false negatives have different impacts on the problem. It provides a single metric that considers both precision and recall, giving a balanced measure of the model's performance.

A high F1-score indicates that the model has both high precision and high recall, meaning it can effectively identify positive instances while minimizing false positives and false negatives.

### Pros and Cons

**Pros:**
- The F1-score provides a single metric that balances both precision and recall.
- It is useful in scenarios where there is an imbalance between classes or when false positives and false negatives have different consequences.
- The F1-score is a robust measure for evaluating model performance, especially in classification tasks.

**Cons:**
- The F1-score does not consider true negatives and may not reflect the overall accuracy of the model.
- It may not be the best choice when the relative importance of precision and recall varies based on the specific problem.

The F1-score should be considered alongside other evaluation metrics, such as accuracy, precision, and recall, to gain a comprehensive understanding of the model's performance in classification tasks.


### Code

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
print("F1-score:", f1)
```

""", unsafe_allow_html=True)
                                




















