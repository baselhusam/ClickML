import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from streamlit_option_menu import option_menu

# Config
st.set_page_config(layout="centered", page_title="Click ML", page_icon="👆", initial_sidebar_state="collapsed")

# Session State
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None

if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None

if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None

if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

if 'X_val' not in st.session_state:
    st.session_state['X_val'] = None

if 'y_val' not in st.session_state:
    st.session_state['y_val'] = None

if "model" not in st.session_state:
    st.session_state['model'] = None

if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = False

if "trained_model_bool" not in st.session_state:
    st.session_state['trained_model_bool'] = False

if "problem_type" not in st.session_state:
    st.session_state['problem_type'] = None

if "metrics_df" not in st.session_state:
    st.session_state['metrics_df'] = pd.DataFrame()

if "is_train" not in st.session_state:
    st.session_state['is_train'] = False

if "is_test" not in st.session_state:
    st.session_state['is_test'] = False

if "is_val" not in st.session_state:
    st.session_state['is_val'] = False

if "show_eval" not in st.session_state:
    st.session_state['show_eval'] = False

metrics_df = pd.DataFrame()

def new_line():
    st.write("\n")

st.cache_data()
def load_data(upd_file):
    df = pd.read_csv(upd_file)
    return df

# Title
col1, col2, col3 = st.columns([0.5,1,0.5])
col2.markdown("<h1 align='center'> 👉 ClickML", unsafe_allow_html=True)

# Subtitle
st.markdown("""<br> Welcome to ClickML, the easy-to-use platform for building machine 
learning models with just a few clicks. Our intuitive interface and powerful tools make it easy to prepare your data, 
train models, and extract insights in minutes, without the need for any prior coding or machine learning knowledge. 
Start building your own models today!""", unsafe_allow_html=True)
st.divider()



# Dataframe selection
st.markdown("<h2 align='center'> Select DataFrame", unsafe_allow_html=True)
new_line()
new_line()

# Upload file
col1, col2, col3 = st.columns([0.9,0.15,0.7])
with col1:
    st.markdown("<h3 align='center'> Upload File", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="csv")

# OR
col2.markdown("<h3>OR", unsafe_allow_html=True)

# Select from ours
with col3:
    st.markdown("<h3 align='center'> Select from Ours", unsafe_allow_html=True)
    selected = st.selectbox("", ["Select", "Titanic Dataset", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", 
                                 "Boston Housing Dataset", "Diabetes Dataset", "Digits Dataset", 
                                 "Olivetti Faces Dataset", "California Housing Dataset", 
                                 "Covid-19 Dataset"])
    

X_train, X_test, y_train, y_test, X_val, y_val = None, None, None, None, None, None


        
with st.sidebar:
    st.markdown("<h2 align='center'> Click ML", unsafe_allow_html=True)
    st.image("./logo2.png",  use_column_width=True)
    

# Dataframe
if uploaded_file is not None or selected != "Select":

    # Load Data
    new_line()
    if st.session_state['df'] is None:
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
        else:
            if selected == "Titanic Dataset":
                df = load_data("./data/titanic.csv")
                st.session_state.df = df
            elif selected == "Iris Dataset":
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['target'] = iris.target
                st.session_state.df = df

            elif selected == "Wine Dataset":
                from sklearn.datasets import load_wine
                wine = load_wine()
                df = pd.DataFrame(wine.data, columns=wine.feature_names)
                df['target'] = wine.target
                st.session_state.df = df

            elif selected == "Breast Cancer Dataset":
                from sklearn.datasets import load_breast_cancer
                cancer = load_breast_cancer()
                df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
                df['target'] = cancer.target
                st.session_state.df = df

            elif selected == "Boston Housing Dataset":
                from sklearn.datasets import load_boston
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['target'] = boston.target
                st.session_state.df = df

            elif selected == "Diabetes Dataset":
                from sklearn.datasets import load_diabetes
                diabetes = load_diabetes()
                df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                df['target'] = diabetes.target
                st.session_state.df = df

            elif selected == "Digits Dataset":
                from sklearn.datasets import load_digits
                digits = load_digits()
                df = pd.DataFrame(digits.data, columns=digits.feature_names)
                df['target'] = digits.target
                st.session_state.df = df

            elif selected == "Olivetti Faces Dataset":
                from sklearn.datasets import fetch_olivetti_faces
                olivetti = fetch_olivetti_faces()
                df = pd.DataFrame(olivetti.data)
                df['target'] = olivetti.target
                st.session_state.df = df

            elif selected == "California Housing Dataset":
                from sklearn.datasets import fetch_california_housing
                california = fetch_california_housing()
                df = pd.DataFrame(california.data, columns=california.feature_names)
                df['target'] = california.target
                st.session_state.df = df

            elif selected == "Covid-19 Dataset":
                df = load_data("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")
                st.session_state.df = df



    df = st.session_state.df
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    X_val = st.session_state.X_val
    y_val = st.session_state.y_val
    trained_model = st.session_state.trained_model
    is_train = st.session_state.is_train
    is_test = st.session_state.is_test
    is_val = st.session_state.is_val
    model = st.session_state.model
    show_eval = st.session_state.show_eval

    st.divider()
    new_line()



    # EDA
    st.markdown("### Exploratory Data Analysis", unsafe_allow_html=True)
    new_line()
    with st.expander("Show EDA"):

        # Head
        new_line()
        head = st.checkbox("Show First 5 Rows", value=False)    
        new_line()
        if head:
            st.write(df.head())

        # Tail
        tail = st.checkbox("Show Last 5 Rows", value=False)
        new_line()
        if tail:
            st.write(df.tail())

        # Shape
        shape = st.checkbox("Show Shape", value=False)
        new_line()
        if shape:
            st.write(f"This DataFrame has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
            new_line()

        # Columns
        columns = st.checkbox("Show Columns", value=False)
        new_line()
        if columns:
            st.write((df.columns).transpose())
            
        # Describe Numerical
        describe = st.checkbox("Show Description (Numerical Features)", value=False)
        new_line()
        if describe:
            st.write(df.describe())

        # Describe Categorical
        describe_cat = st.checkbox("Show Description (Categorical Features)", value=False)
        new_line()
        if describe_cat:
            if df.select_dtypes(include=np.object).columns.tolist():
                st.write(df.describe(include=['object']))
            else:
                st.info("There is no more categorical Features.")

        # Missing Values
        missing = st.checkbox("Show Missing Values", value=False)
        new_line()
        if missing:
            col1, col2 = st.columns([0.4,1])
            col1.write(df.isnull().sum())
            with col2:
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values = df.isnull().sum()
                null_values = null_values[null_values > 0]
                null_values = null_values.sort_values(ascending=False)
                null_values = null_values.to_frame()
                null_values.columns = ['Count']
                null_values.index.names = ['Feature']
                null_values['Feature'] = null_values.index
                fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                st.plotly_chart(fig, use_container_width=True)
                 

        # Delete Columns
        delete = st.checkbox("Delete Columns", value=False)
        new_line()
        if delete:
            col_to_delete = st.multiselect("Select Columns to Delete", df.columns)
            
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Delete"):
                new_line()
                df.drop(columns=col_to_delete, inplace=True)
                st.session_state.df = df
                st.success("Columns Deleted Successfully!")
                new_line()

        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame")
        new_line()

        if show_df:
            st.write(df)
                


    # Missing Values
    new_line()
    st.markdown("### Missing Values", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Missing Values"):

        # INFO
        new_line()
        missing = st.checkbox("Further Analysis", value=False, key='missing')
        new_line()
        if missing:

            col1, col2 = st.columns(2)
            with col1:
                # Number of Null Values
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum(), width=400, height=300)

            with col2:
                # Percentage of Null Values
                st.markdown("<h6 align='center'> Percentage of Null Values", unsafe_allow_html=True)
                null_percentage = pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                null_percentage.columns = ['Percentage']
                null_percentage = null_percentage.sort_values(by='Percentage', ascending=False)
                st.dataframe(null_percentage, width=400, height=300)

            # Heatmap
            col1, col2, col3 = st.columns([0.1,1,0.1])
            with col2:
                new_line()
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values = df.isnull().sum()
                null_values = null_values[null_values > 0]
                null_values = null_values.sort_values(ascending=False)
                null_values = null_values.to_frame()
                null_values.columns = ['Count']
                null_values.index.names = ['Feature']
                null_values['Feature'] = null_values.index
                fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                st.plotly_chart(fig, use_container_width=True)


        # INPUT
        col1, col2 = st.columns(2)
        with col1:
            fill_feat = st.multiselect("Select Features", df.columns, help="Select Features to fill missing values")

        with col2:
            strategy = st.selectbox("Select Missing Values Strategy", ["Select", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode (Most Frequent)", "Fill with ffill, bfill"], help="Select Missing Values Strategy")


        if fill_feat and strategy != "Select":

            new_line()
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Apply", key="missing_apply", help="Apply Missing Values Strategy"):
                
                # Drop Rows
                if strategy == "Drop Rows":
                    df = df.dropna(axis=0)
                    st.session_state['df'] = df
                    st.success("Missing values have been dropped from the DataFrame.")
                    new_line()


                # Drop Columns
                elif strategy == "Drop Columns":
                    df = df.dropna(axis=1)
                    st.session_state['df'] = df
                    st.success("Missing values have been dropped from the DataFrame.")
                    new_line()


                # Fill with Mean
                elif strategy == "Fill with Mean":
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='mean')
                    df[fill_feat] = num_imputer.fit_transform(df[fill_feat])

                    if df.select_dtypes(include=np.object).columns.tolist():
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        cat_feat = df.select_dtypes(include=np.object).columns.tolist()
                        df[cat_feat] = cat_imputer.fit_transform(df[cat_feat])

                    st.session_state['df'] = df
                    st.success("Missing values have been filled with the mean of the respective column.")
                    

                # Fill with Median
                elif strategy == "Fill with Median":
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='median')
                    df[fill_feat] = pd.DataFrame(num_imputer.fit_transform(df[fill_feat]), columns=df[fill_feat].columns)

                    if df.select_dtypes(include=np.object).columns.tolist():
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        cat_feat = df.select_dtypes(include=np.object).columns.tolist()
                        df[cat_feat] = cat_imputer.fit_transform(df[cat_feat])

                    st.session_state['df'] = df
                    st.success("Missing values have been filled with the median of the respective column.")


                # Fill with Mode
                elif strategy == "Fill with Mode (Most Frequent)":
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[fill_feat] = imputer.fit_transform(df[fill_feat])

                    st.session_state['df'] = df
                    st.success("Missing values have been filled with the mode of the respective column.")


                # Fill with ffill, bfill
                elif strategy == "Fill with ffill, bfill":
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    st.session_state['df'] = df
                    st.success("Missing values have been filled with the ffill and bfill method.")
        
        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="missing_show_df")
        new_line()
        if show_df:
            st.write(df)


    # Encoding
    new_line()
    st.markdown("### Handling Categorical Data", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Encoding"):
        new_line()

        # INFO
        show_cat = st.checkbox("Categorical features are:")
        if show_cat:
            st.dataframe(df.select_dtypes(include=np.object), height=250, )

        further_analysis = st.checkbox("Further Analysis", value=False, key='further_analysis')
        if further_analysis:

            col1, col2 = st.columns([0.5,1])
            with col1:
                # Each categorical feature has how many unique values as dataframe
                new_line()
                st.markdown("<h6 align='left'> Number of Unique Values", unsafe_allow_html=True)
                unique_values = pd.DataFrame(df.select_dtypes(include=np.object).nunique())
                unique_values.columns = ['# Unique Values']
                unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                st.dataframe(unique_values, width=200, height=300)

            with col2:
                # Plot for the count of unique values for the categorical features
                new_line()
                st.markdown("<h6 align='center'> Plot for the Count of Unique Values ", unsafe_allow_html=True)
                unique_values = pd.DataFrame(df.select_dtypes(include=np.object).nunique())
                unique_values.columns = ['# Unique Values']
                unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                unique_values['Feature'] = unique_values.index
                fig = px.bar(unique_values, x='Feature', y='# Unique Values', color='# Unique Values', height=350)
                st.plotly_chart(fig, use_container_width=True)


        # Explain
        exp_enc = st.checkbox("Explain Encoding", value=False, key='exp_enc')
        if exp_enc:
            new_line()
            col1, col2 = st.columns([0.8,1])
            with col1:
                st.markdown("<h6 align='center'>Ordinal Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns(2)
                with colb:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']) ),width=120, height=200)
                with cola:
                    st.write("After Encoding")
                    st.dataframe(pd.DataFrame(np.array([0,1,2,1,0])),width=120, height=200)

            with col2:
                st.markdown("<h6 align='center'>One Hot Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns([0.7,1])
                with cola:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a','b','c', 'b','a']) ),width=150, height=200)
                with colb:
                    st.write("After Encoding")
                    st.dataframe(pd.DataFrame(np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,0],[1,0,0]])),width=200, height=200)

            col1, col2, col3 = st.columns([0.5,1,0.5])
            with col2:
                new_line()
                st.markdown("<h6 align='center'>Count Frequency Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns([0.8,1])
                with cola:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a','b','c', 'b','a']) ),width=150, height=200)
                with colb:
                    st.write("After Encoding")
                    st.dataframe(pd.DataFrame(np.array([0.4,0.4,0.2,0.4,0.4])),width=200, height=200)


        # INPUT
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            enc_feat = st.multiselect("Select Features", df.select_dtypes(include=np.object).columns.tolist(), key='encoding_feat', help="Select the categorical features to encode.")

        with col2:
            encoding = st.selectbox("Select Encoding", ["Select", "Ordinal Encoding", "One Hot Encoding", "Count Frequency Encoding"], key='encoding', help="Select the encoding method.")


        if enc_feat and encoding != "Select":
            col1, col2, col3 = st.columns([1,0.5,1])
            new_line()
            if col2.button("Apply", key='encoding_apply',use_container_width=True ,help="Click to apply encoding."):
                # Ordinal Encoding
                new_line()
                if encoding == "Ordinal Encoding":
                    from sklearn.preprocessing import OrdinalEncoder
                    encoder = OrdinalEncoder()
                    cat_cols = enc_feat
                    df[cat_cols] = encoder.fit_transform(df[cat_cols])
                    st.session_state['df'] = df
                    st.success("Categorical features have been encoded using Ordinal Encoding.")
                    
                # One Hot Encoding
                elif encoding == "One Hot Encoding":
                    df = pd.get_dummies(df, columns=enc_feat)
                    st.session_state['df'] = df
                    st.success("Categorical features have been encoded using One Hot Encoding.")

                # Count Frequency Encoding
                elif encoding == "Count Frequency Encoding":
                    df[enc_feat] = df[enc_feat].apply(lambda x: x.map(len(df) / x.value_counts()))
                    st.session_state['df'] = df
                    st.success("Categorical features have been encoded using Count Frequency Encoding.")

        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="cat_show_df", help="Click to show the DataFrame.")
        new_line()
        if show_df:
            st.write(df)


    # Scaling
    new_line()
    st.markdown("### Scaling", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Scaling"):
        new_line()



        # Ranges for the numeric features
        feat_range = st.checkbox("Show Values Ranges for each Numerical Feature", value=False, key='feat_range')
        if feat_range:
            new_line()
            col1, col2, col3 = st.columns([0.05,1, 0.05])
            with col2:
                 st.dataframe(df.describe().T, width=700)
            
            new_line()



        # Scaling Methods
        scaling_methods = st.checkbox("Don't sure what is scaling? Further Information", value=False, key='scaling_methods')
        if scaling_methods:
            new_line()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h6 align='center'> Standard Scaling </h6>" ,unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - \mu}{\sigma}''')
                new_line()
                # Values Ranges for the output of Standard Scaling in general
                st.latex(r'''z \in [-3,3]''')   

            with col2:
                st.markdown("<h6 align='center'> MinMax Scaling </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - min(x)}{max(x) - min(x)}''')
                new_line()
                # Values Ranges for the output of MinMax Scaling in general
                st.latex(r'''z \in [0,1]''')
                
            with col3:
                st.markdown("<h6 align='center'> Robust Scaling </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - Q_1}{Q_3 - Q_1}''')
                # Values Ranges for the output of Robust Scaling in general
                new_line()
                st.latex(r'''z \in [-2,2]''')



        # INPUT
        new_line()
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            scale_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features to be scaled.")

        with col2:
            scaling = st.selectbox("Select Scaling", ["Select", "Standard Scaling", "MinMax Scaling", "Robust Scaling"], help="Select the scaling method.")


        if scale_feat and scaling != "Select":       
                col1, col2, col3 = st.columns([1, 0.5, 1])
                new_line()

                if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling."):
                    new_line()
    
                    # Standard Scaling
                    if scaling == "Standard Scaling":
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success("Numerical features have been scaled using Standard Scaling.")
    
                    # MinMax Scaling
                    elif scaling == "MinMax Scaling":
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success("Numerical features have been scaled using MinMax Scaling.")
    
                    # Robust Scaling
                    elif scaling == "Robust Scaling":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success("Numerical features have been scaled using Robust Scaling.")

        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="scaling_show_df", help="Click to show the DataFrame.")
        new_line()
        if show_df:
            st.write(df)


    # Data Transformation
    new_line()
    st.markdown("### Data Transformation", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Transformation"):
        new_line()
        


        # Transformation Methods
        trans_methods = st.checkbox("Further Information About Transformation Methods", key="trans_methods", value=False)
        if trans_methods:
            new_line()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("<h6 align='center'> Log <br> Transformation</h6>", unsafe_allow_html=True)
                st.latex(r'''z = log(x)''')

            with col2:
                st.markdown("<h6 align='center'> Square Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt{x}''')

            with col3:
                st.markdown("<h6 align='center'> Cube Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt[3]{x}''')

            with col4:
                st.markdown("<h6 align='center'> Exponential Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = e^x''')



        # INPUT
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            trans_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features you want to transform.", key="transformation features")

        with col2:
            trans = st.selectbox("Select Transformation", ["Select", "Log Transformation", "Square Root Transformation", "Cube Root Transformation", "Exponential Transformation"],
                                  help="Select the transformation you want to apply.", 
                                  key= "transformation")
        

        if trans_feat and trans != "Select":
            new_line()
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Apply", key='trans_apply',use_container_width=True ,help="Click to apply transformation."):
                new_line()
                # Log Transformation
                if trans == "Log Transformation":
                    df[trans_feat] = np.log1p(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Log Transformation.")

                # Square Root Transformation
                elif trans == "Square Root Transformation":
                    df[trans_feat] = np.sqrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Square Root Transformation.")

                # Cube Root Transformation
                elif trans == "Cube Root Transformation":
                    df[trans_feat] = np.cbrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Cube Root Transformation.")

                # Exponential Transformation
                elif trans == "Exponential Transformation":
                    df[trans_feat] = np.exp(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Exponential Transformation.")

        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="trans_show_df", help="Click to show the DataFrame.")
        new_line()
        if show_df:
            st.write(df)


    # Feature Engineering
    new_line()
    st.markdown("### Feature Engineering", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Feature Engineering"):

        # Feature Extraction
        new_line()
        st.markdown("#### Feature Extraction", unsafe_allow_html=True)
        new_line()

        col1, col2, col3 = st.columns(3)
        with col1:  
            feat1 = st.multiselect("First Feature/s", df.select_dtypes(include=np.number).columns.tolist(), key="feat_ex1", help="Select the first feature/s you want to extract.")
        with col2:
            op = st.selectbox("Mathematical Operation", ["Select", "Addition +", "Subtraction -", "Multiplication *", "Division /"], key="feat_ex_op", help="Select the mathematical operation you want to apply.")
        with col3:
            feat2 = st.multiselect("Second Feature/s", df.select_dtypes(include=np.number).columns.tolist(), key="feat_ex2", help="Select the second feature/s you want to extract.")

        if feat1 and op != "Select" and feat2:
            col1, col2, col3 = st.columns(3)
            with col2:
                feat_name = st.text_input("Feature Name", key="feat_name", help="Enter the name of the new feature.")

            col1, col2, col3 = st.columns([1, 0.6, 1])
            new_line()
            if col2.button("Extract Feature"):
                if feat_name == "":
                    feat_name = f"({feat1[0]} {op} {feat2[0]})"

                if op == "Addition +":
                    df[feat_name] = df[feat1[0]] + df[feat2[0]]
                    st.session_state['df'] = df
                    st.success(f"Feature '**_{feat_name}_**' has been extracted using Addition.")

                elif op == "Subtraction -":
                    df[feat_name] = df[feat1[0]] - df[feat2[0]]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Subtraction.")

                elif op == "Multiplication *":
                    df[feat_name] = df[feat1[0]] * df[feat2[0]]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Multiplication.")

                elif op == "Division /":
                    df[feat_name] = df[feat1[0]] / df[feat2[0]]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Division.")



        # Feature Transformation
        st.divider()
        st.markdown("#### Feature Transformation", unsafe_allow_html=True)
        new_line()

        col1, col2, col3 = st.columns(3)
        with col1:    
            feat_trans = st.multiselect("Select Feature/s", df.select_dtypes(include=np.number).columns.tolist(), help="Select the Features you want to Apply transformation operation on it")
        with col2:
            op = st.selectbox("Select Operation", ["Select", "Addition +", "Subtraction -", "Multiplication *", "Division /", ], key='feat_trans_op', help="Select the operation you want to apply on the feature")
        with col3:
            value = st.text_input("Enter Value", key='feat_trans_val', help="Enter the value you want to apply the operation on it")

        

        if op != "Select" and value != "":
            col1, col2, col3 = st.columns([1, 0.7, 1])
            new_line()
            if col2.button("Transform Feature"):
                if op == "Addition +":
                    df[feat_trans] = df[feat_trans] + float(value)
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Addition.")

                elif op == "Subtraction -":
                    df[feat_trans] = df[feat_trans] - float(value)
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Subtraction.")

                elif op == "Multiplication *":
                    df[feat_trans] = df[feat_trans] * float(value)
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Multiplication.")

                elif op == "Division /":
                    df[feat_trans] = df[feat_trans] / float(value)
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Division.")



        # Feature Selection
        st.divider()
        st.markdown("#### Feature Selection", unsafe_allow_html=True)
        new_line()

        feat_sel = st.multiselect("Select Feature/s", df.columns.tolist(), key='feat_sel', help="Select the Features you want to keep in the dataset")
        new_line()

        if feat_sel:
            col1, col2, col3 = st.columns([1, 0.7, 1])
            if col2.button("Select Features"):
                new_line()
                df = df[feat_sel]
                st.session_state['df'] = df
                st.success("Features have been selected.")
        
        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="feat_eng_show_df", help="Click to show the DataFrame.")
        new_line()
        if show_df:
            st.write(df)


    # Data Splitting
    new_line()
    st.markdown("### Data Splitting", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Splitting"):

        new_line()
        train_size, val_size, test_size = 0,0,0
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
        with col2:
            sets = st.multiselect("Select The Split Sets", ["Train", "Validation", "Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model. ")

        st.session_state['is_train'] = True if "Train" in sets else False
        st.session_state['is_val'] = True if "Validation" in sets else False
        st.session_state['is_test'] = True if "Test" in sets else False

        if sets and target:
            if ["Train", "Validation", "Test"] == sets or ["Validation", "Train", "Test"] == sets or ["Test", "Train", "Validation"] == sets or ["Train", "Test", "Validation"] == sets or ["Validation", "Test", "Train"] == sets or ["Test", "Validation", "Train"] == sets:
                new_line()
                col1, col2, col3 = st.columns(3)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                with col2:
                    val_size = st.number_input("Validation Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='val_size')
                with col3:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='test_size')

                if float(train_size + val_size + test_size) != 1.0:
                    new_line()
                    st.error("The sum of Train, Validation, and Test sizes must be equal to 1.0")
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data")
                        
                        if split_button:
                            from sklearn.model_selection import train_test_split
                            X_train, X_rem, y_train, y_rem = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= val_size / (1.0 - train_size),random_state=42)
                            st.session_state['X_train'] = X_train
                            st.session_state['X_val'] = X_val
                            st.session_state['X_test'] = X_test
                            st.session_state['y_train'] = y_train
                            st.session_state['y_val'] = y_val
                            st.session_state['y_test'] = y_test

                    
                    col1, col2, col3 = st.columns(3)
                    if split_button:
                        st.success("Data Splitting Done!")
                        with col1:
                            st.write("Train Set")
                            st.write("X Train Shape: ", X_train.shape)
                            st.write("Y Train Shape: ", y_train.shape)

                            train = pd.concat([X_train, y_train], axis=1)
                            train_csv = train.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key='train3')

                        with col2:
                            st.write("Validation Set")
                            st.write("X Validation Shape: ", X_val.shape)
                            st.write("Y Validation Shape: ", y_val.shape)

                            val = pd.concat([X_val, y_val], axis=1)
                            val_csv = val.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Validation Set", val_csv, "validation.csv", key='val3')

                        with col3:
                            st.write("Test Set")
                            st.write("X Test Shape: ", X_test.shape)
                            st.write("Y Test Shape: ", y_test.shape)

                            test = pd.concat([X_test, y_test], axis=1)
                            test_csv = test.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Test Set", test_csv, "test.csv", key='test3')


            elif ["Train", "Validation"] == sets or ["Validation", "Train"] == sets:

                new_line()
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                with col2:
                    val_size = st.number_input("Validation Size", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='val_size')

                if float(train_size + val_size) != 1.0:
                    new_line()
                    st.error("The sum of Train and Validation sizes must be equal to 1.0")
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data")

                        if split_button:
                            from sklearn.model_selection import train_test_split
                            X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                            st.session_state['X_train'] = X_train
                            st.session_state['X_val'] = X_val
                            st.session_state['y_train'] = y_train
                            st.session_state['y_val'] = y_val

                    
                    col1, col2 = st.columns(2)
                    if split_button:
                        st.success("Data Splitting Done!")
                        with col1:
                            st.write("Train Set")
                            st.write("X Train Shape: ", X_train.shape)
                            st.write("Y Train Shape: ", y_train.shape)

                            train = pd.concat([X_train, y_train], axis=1)
                            train_csv = train.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Train Set", train_csv, "train.csv", key='train2')

                        with col2:
                            st.write("Validation Set")
                            st.write("X Validation Shape: ", X_val.shape)
                            st.write("Y Validation Shape: ", y_val.shape)

                            val = pd.concat([X_val, y_val], axis=1)
                            val_csv = val.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Validation Set", val_csv, "validation.csv", key='val2')

            elif ["Train", "Test"] == sets or ["Test", "Train"] == sets:
                    
                    new_line()
                    col1, col2 = st.columns(2)
                    with col1:
                        train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    with col2:
                        test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='test_size')
    
                    if float(train_size + test_size) != 1.0:
                        new_line()
                        st.error("The sum of Train and Test sizes must be equal to 1.0")
                        new_line()
    
                    else:
                        split_button = ""
                        col1, col2, col3 = st.columns([1, 0.5, 1])
                        with col2:
                            new_line()
                            split_button = st.button("Split Data")
    
                            if split_button:
                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                                st.session_state['X_train'] = X_train
                                st.session_state['X_test'] = X_test
                                st.session_state['y_train'] = y_train
                                st.session_state['y_test'] = y_test
    
                        
                        col1, col2 = st.columns(2)
                        if split_button:
                            st.success("Data Splitting Done!")
                            with col1:
                                st.write("Train Set")
                                st.write("X Train Shape: ", X_train.shape)
                                st.write("Y Train Shape: ", y_train.shape)

                                train = pd.concat([X_train, y_train], axis=1)
                                train_csv = train.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Train Set", train_csv, "train.csv", key='train1')

    
                            with col2:
                                st.write("Test Set")
                                st.write("X Test Shape: ", X_test.shape)
                                st.write("Y Test Shape: ", y_test.shape)

                                test = pd.concat([X_test, y_test], axis=1)
                                test_csv = test.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Test Set", test_csv, "test.csv", key='test1')

            elif "Test" in sets and "Validation" in sets:
                st.error("There must be a Train set to split the data")
            else:
                new_line()
                st.error("It must have at least two sets to split the data")
                new_line()


    # Building the model
    new_line()
    st.markdown("### Building the Model")
    new_line()
    problem_type = ""
    with st.expander("Model Building"):    
        
        target, problem_type, model = "", "", ""
        col1, col2, col3 = st.columns(3)

        with col1:
            target = st.selectbox("Target Variable", ["Select"] + df.columns.tolist(), key='target_ml', help="The target variable is the variable that you want to predict")
            new_line()

        with col2:
            problem_type = st.selectbox("Problem Type", ["Classification", "Regression"], key='problem_type', help="The problem type is the type of problem that you want to solve")

        with col3:

            if problem_type == "Classification":
                model = st.selectbox("Model", ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()

            elif problem_type == "Regression":
                model = st.selectbox("Model", ["Linear Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()


        if target != "Select" and problem_type and model:
            
            if problem_type == "Classification":
                 
                # Hyperparameters Tuning for each model
                if model == "Logistic Regression":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        penalty = st.selectbox("Penalty (Optional)", ["l2", "l1", "none", "elasticnet"], key='penalty')

                    with col2:
                        solver = st.selectbox("Solver (Optional)", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"], key='solver')

                    with col3:
                        C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')

                    
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True

                        # Train the model
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(penalty=penalty, solver=solver, C=C, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors **Required**", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')

                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')

                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')

                    
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True

                        # Train the model
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "Support Vector Machine":
                        
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["rbf", "poly", "linear", "sigmoid", "precomputed"], key='kernel')
    
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=100, value=3, step=1, key='degree')
    
                    with col3:
                        C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')
    
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
    
                        # Train the model
                        from sklearn.svm import SVC
                        model = SVC(kernel=kernel, degree=degree, C=C, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")
    
                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "Decision Tree":
                            
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
        
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
        
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                            
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
        
                        # Train the model
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "Random Forest":
                                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                                
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "XGBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
            
                    with col3:
                        booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from xgboost import XGBClassifier
                        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, booster=booster, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == 'LightGBM':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["gbdt", "dart", "goss", "rf"], key='boosting_type')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from lightgbm import LGBMClassifier
                        model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == 'CatBoost':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["Ordered", "Plain"], key='boosting_type')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from catboost import CatBoostClassifier
                        model = CatBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type, random_state=42)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')      

            if problem_type == "Regression":
                 
                if model == "Linear Regression":
                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fit_intercept = st.selectbox("Fit Intercept (Optional)", [True, False], key='normalize')
            
                    with col2:
                        positive = st.selectbox("Positve (Optional)", [True, False], key='positive')
            
                    with col3:
                        copy_x = st.selectbox("Copy X (Optional)", [True, False], key='copy_x')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression(fit_intercept=fit_intercept, positive=positive, copy_X=copy_x)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors (Optional)", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')
            
                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')
            
                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "Support Vector Machine":
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["linear", "poly", "rbf", "sigmoid", "precomputed"], key='kernel')
            
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=10, value=3, step=1, key='degree')
            
                    with col3:
                        gamma = st.selectbox("Gamma (Optional)", ["scale", "auto"], key='gamma')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from sklearn.svm import SVR
                        model = SVR(kernel=kernel, degree=degree, gamma=gamma)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "Decision Tree":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
            
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=10, value=2, step=1, key='min_samples_split')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')
                
                if model == "Random Forest":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=10, value=2, step=1, key='min_samples_split')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "XGBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from xgboost import XGBRegressor
                        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, booster=booster)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                if model == "LightGBM":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["gbdt", "dart", "goss", "rf"], key='boosting_type')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from lightgbm import LGBMRegressor
                        model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model') 

                if model == "CatBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["Ordered", "Plain"], key='boosting_type')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type)
                        model.fit(X_train, y_train)
                        st.session_state['trained_model'] = model
                        st.success("Model Trained Successfully!")

                        # save the model
                        import joblib
                        joblib.dump(model, 'model.pkl')

                        # Download the model
                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')


    # Evaluation
    if st.session_state['trained_model_bool']:
        st.markdown("## Evaluation")
        new_line()
        with st.expander("Model Evaluation"):
            # Load the model
            import joblib
            model = joblib.load('model.pkl')

            # Predictions
            if st.session_state['y_test'] is not None:
                y_pred = model.predict(X_test)

            if st.session_state['y_train'] is not None:
                y_pred_train = model.predict(X_train)

            if st.session_state['y_val'] is not None:
                y_pred_val = model.predict(X_val)

            # Choose Evaluation Metric
            if st.session_state['problem_type'] == "Classification":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"], key='evaluation_metric')

            elif st.session_state['problem_type'] == "Regression":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2 Score"], key='evaluation_metric')

            col1, col2, col3 = st.columns([1, 0.6, 1])
            if col2.button("Evaluate Model"):
                st.session_state.show_eval = True
                # Make the Evalutaion and store the results in a DataFrame in session state for st.session_state['metrics_df'] for the test and the training sets
                for metric in evaluation_metric:
                    
                    if metric == "Accuracy":
                        from sklearn.metrics import accuracy_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None:
                            test_score = accuracy_score(y_test, y_pred)
                        if st.session_state['y_train'] is not None :
                            train_score = accuracy_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            val_score = accuracy_score(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "Precision":
                        from sklearn.metrics import precision_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None :
                            test_score = precision_score(y_test, y_pred)
                        if st.session_state['y_train'] is not None :
                            train_score = precision_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            val_score = precision_score(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "Recall":
                        from sklearn.metrics import recall_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None :
                            test_score = recall_score(y_test, y_pred)
                        if st.session_state['y_train'] is not None:
                            train_score = recall_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None:
                            val_score = recall_score(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "F1 Score":
                        from sklearn.metrics import f1_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None:
                            test_score = f1_score(y_test, y_pred)
                        if st.session_state['y_train'] is not None:
                            train_score = f1_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None:
                            val_score = f1_score(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "AUC Score":
                        from sklearn.metrics import roc_auc_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None:
                            test_score = roc_auc_score(y_test, y_pred)
                        if st.session_state['y_train'] is not None:
                            train_score = roc_auc_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            val_score = roc_auc_score(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df
                        
                    elif metric == "Mean Absolute Error (MAE)":
                        from sklearn.metrics import mean_absolute_error
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None :
                            test_score = mean_absolute_error(y_test, y_pred)
                        if st.session_state['y_train'] is not None :
                            train_score = mean_absolute_error(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            val_score = mean_absolute_error(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df


                    elif metric == "Mean Squared Error (MSE)":
                        from sklearn.metrics import mean_squared_error
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None :
                            test_score = mean_squared_error(y_test, y_pred)
                        if st.session_state['y_train'] is not None :
                            train_score = mean_squared_error(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            val_score = mean_squared_error(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df


                    elif metric == "Root Mean Squared Error (RMSE)":
                        from sklearn.metrics import mean_squared_error
                        from math import sqrt
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None :
                            test_score = sqrt(mean_squared_error(y_test, y_pred))
                        if st.session_state['y_train'] is not None :
                            train_score = sqrt(mean_squared_error(y_train, y_pred_train))
                        if st.session_state['y_val'] is not None :
                            val_score = sqrt(mean_squared_error(y_val, y_pred_val))

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df
                        
                        

                    elif metric == "R2 Score":
                        from sklearn.metrics import r2_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_test'] is not None :
                            test_score = r2_score(y_test, y_pred)
                        if st.session_state['y_train'] is not None :
                            train_score = r2_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            val_score = r2_score(y_val, y_pred_val)

                        lst = [test_score, train_score, val_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df


            # Show Evaluation Metric
            if show_eval:
                new_line()
                col1, col2, col3 = st.columns([0.5, 1, 0.5])
                st.markdown("### Evaluation Metric")

                if is_train and is_test and is_val:
                    st.session_state['metrics_df'].index = ['Test', 'Train', 'Validation']
                    st.write(st.session_state['metrics_df'])

                elif is_train and is_test:
                    st.session_state['metrics_df'].index = ['Test', 'Train']
                    st.write(st.session_state['metrics_df'])

                elif is_train and is_val:
                    st.session_state['metrics_df'].index = ['Train', 'Validation']
                    st.write(st.session_state['metrics_df'])


                # Show Evaluation Metric Plot
                new_line()
                st.markdown("### Evaluation Metric Plot")
                st.line_chart(st.session_state['metrics_df'])
                        
                # Show Confusion Matrix as plot
                if st.session_state['problem_type'] == "Classification":
                    from sklearn.metrics import plot_confusion_matrix
                    st.markdown("### Confusion Matrix")
                    new_line()
                    
                    if is_test:
                        # Show the confusion matrix plot without any columns
                        col1, col2, col3 = st.columns([0.2, 1, 0.2])
                        fig, ax = plt.subplots()
                        plot_confusion_matrix(model, X_test, y_test, ax=ax)
                        col2.pyplot(fig)

                    elif is_val:
                        # Show the confusion matrix plot without any columns
                        col1, col2, col3 = st.columns([0.2, 1, 0.2])
                        fig, ax = plt.subplots()
                        plot_confusion_matrix(model, X_val, y_val, ax=ax)
                        col2.pyplot(fig)
                
                
                    
    # Show DataFrame Button
    new_line()
    col1, col2, col3 = st.columns([1, 0.6, 1])
    with col2:
        show_df = st.button("Show DataFrame", key='show_df_general')
    new_line()

    if show_df:
        st.write(df)

