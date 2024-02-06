# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image

# Config
page_icon = Image.open("./assets/logoo.png")
st.set_page_config(layout="centered", page_title="Complaints-AI", page_icon=page_icon)

# Initial State
def initial_state():
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

    if "all_the_process" not in st.session_state:
        st.session_state['all_the_process'] = """"""

    if "all_the_process_predictions" not in st.session_state:
        st.session_state['all_the_process_predictions'] = False

    if 'y_pred_train' not in st.session_state:
        st.session_state['y_pred_train'] = None

    if 'y_pred_test' not in st.session_state:
        st.session_state['y_pred_test'] = None

    if 'y_pred_val' not in st.session_state:
        st.session_state['y_pred_val'] = None

    if 'uploading_way' not in st.session_state:
        st.session_state['uploading_way'] = None

    if "lst_models" not in st.session_state:
        st.session_state["lst_models"] = []

    if "lst_models_predctions" not in st.session_state:
        st.session_state["lst_models_predctions"] = []

    if "models_with_eval" not in st.session_state:
        st.session_state["models_with_eval"] = dict()

    if "reset_1" not in st.session_state:
        st.session_state["reset_1"] = False

initial_state()

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

# Load Data
st.cache_data()
def load_data(upd_file):
    df = pd.read_csv(upd_file)
    return df

# Progress Bar
def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)


# Logo 
col1, col2, col3 = st.columns([0.25,1,0.25])
col2.image("./assests/logoo.png", use_column_width=True)
new_line(2)

# Description
st.markdown("""Welcome to Complaints-AI, the easy-to-use platform for building machine 
learning models with just a few clicks. Our intuitive interface and powerful tools make it easy to prepare your data, 
train models, and extract insights in minutes, without the need for any prior coding or machine learning knowledge. 
Start building your own models today!""", unsafe_allow_html=True)
st.divider()


# Dataframe selection
st.markdown("<h2 align='center'> <b> Getting Started", unsafe_allow_html=True)
new_line(1)
st.write("The first step is to upload your data. You can upload your data by browsing your computer : **Upload File** . The data should be a csv file and should not exceed 200 MB.")
new_line(1)



# Uploading Way
uploading_way = st.session_state.uploading_way
col1 = st.columns(1,gap='large')

# Upload
def upload_click(): st.session_state.uploading_way = "upload"
col1.markdown("<h5 align='center'> Upload File", unsafe_allow_html=True)
col1.button("Upload File", key="upload_file", use_container_width=True, on_click=upload_click)

# No Data
if st.session_state.df is None:

    # Upload
    if uploading_way == "upload":
        uploaded_file = st.file_uploader("Upload the Dataset", type="csv")
        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state.df = df


# Sidebar       
with st.sidebar:
    st.image("./assets/sb-click.png",   use_column_width=True)
    
    
# Dataframe
if st.session_state.df is not None:

    # Re-initialize the variables from the state
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
    y_pred_train = st.session_state.y_pred_train
    y_pred_test = st.session_state.y_pred_test
    y_pred_val = st.session_state.y_pred_val
    metrics_df = st.session_state.metrics_df

    st.divider()
    new_line()


    # EDA
    st.markdown("### 🕵️‍♂️ Exploratory Data Analysis", unsafe_allow_html=True)
    new_line()
    with st.expander("Show EDA"):
        new_line()

        # Head
        head = st.checkbox("Show First 5 Rows", value=False)    
        new_line()
        if head:
            st.dataframe(df.head(), use_container_width=True)

        # Tail
        tail = st.checkbox("Show Last 5 Rows", value=False)
        new_line()
        if tail:
            st.dataframe(df.tail(), use_container_width=True)

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
            st.write(pd.DataFrame(df.columns, columns=['Columns']).T)
            new_line()

            
        # Describe Numerical
        describe = st.checkbox("Show Description **(Numerical Features)**", value=False)
        new_line()
        if describe:
            st.dataframe(df.describe(), use_container_width=True)
            new_line()

        # Describe Categorical
        describe_cat = st.checkbox("Show Description **(Categorical Features)**", value=False)
        new_line()
        if describe_cat:
            if df.select_dtypes(include=np.object).columns.tolist():
                st.dataframe(df.describe(include=['object']), use_container_width=True)
                new_line()
            else:
                st.info("There is no Categorical Features.")
                new_line()

        # Correlation Matrix using heatmap seabron
        corr = st.checkbox("Show Correlation", value=False)
        new_line()
        if corr:

            if df.corr().columns.tolist():
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), cmap='Blues', annot=True, ax=ax)
                st.pyplot(fig)
                new_line()
            else:
                st.info("There is no Numerical Features.")
            

        # Missing Values
        missing = st.checkbox("Show Missing Values", value=False)
        new_line()
        if missing:

            col1, col2 = st.columns([0.4,1])
            with col1:
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False),height=350, use_container_width=True)

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

            new_line()
                 

        # Delete Columns
        delete = st.checkbox("Delete Columns", value=False)
        new_line()
        if delete:
            col_to_delete = st.multiselect("Select Columns to Delete", df.columns)
            new_line()
            
            col1, col2, col3 = st.columns([1,0.7,1])
            if col2.button("Delete", use_container_width=True):
                st.session_state.all_the_process += f"""
# Delete Columns
df.drop(columns={col_to_delete}, inplace=True)
\n """
                progress_bar()
                df.drop(columns=col_to_delete, inplace=True)
                st.session_state.df = df
                st.success(f"The Columns **`{col_to_delete}`** are Deleted Successfully!")


        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        if col2.button("Show DataFrame", use_container_width=True):
            st.dataframe(df, use_container_width=True)
        

    # Missing Values
    new_line()
    st.markdown("### ⚠️ Missing Values", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Missing Values"):

        # Further Analysis
        new_line()
        missing = st.checkbox("Further Analysis", value=False, key='missing')
        new_line()
        if missing:

            col1, col2 = st.columns(2, gap='medium')
            with col1:
                # Number of Null Values
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False), height=300, use_container_width=True)

            with col2:
                # Percentage of Null Values
                st.markdown("<h6 align='center'> Percentage of Null Values", unsafe_allow_html=True)
                null_percentage = pd.DataFrame(round(df.isnull().sum()/df.shape[0]*100, 2))
                null_percentage.columns = ['Percentage']
                null_percentage['Percentage'] = null_percentage['Percentage'].map('{:.2f} %'.format)
                null_percentage = null_percentage.sort_values(by='Percentage', ascending=False)
                st.dataframe(null_percentage, height=300, use_container_width=True)

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
            missing_df_cols = df.columns[df.isnull().any()].tolist()
            if missing_df_cols:
                add_opt = ["All Numerical Features (ClickML Feature)", "All Categorical Feature (ClickML Feature)"]
            else:
                add_opt = []
            fill_feat = st.multiselect("Select Features",  missing_df_cols + add_opt ,  help="Select Features to fill missing values")

        with col2:
            strategy = st.selectbox("Select Missing Values Strategy", ["Select", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode (Most Frequent)", "Fill with ffill, bfill"], help="Select Missing Values Strategy")


        if fill_feat and strategy != "Select":

            new_line()
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Apply", use_container_width=True, key="missing_apply", help="Apply Missing Values Strategy"):

                progress_bar()
                
                # All Numerical Features
                if "All Numerical Features (ClickML Feature)" in fill_feat:
                    fill_feat.remove("All Numerical Features (ClickML Feature)")
                    fill_feat += df.select_dtypes(include=np.number).columns.tolist()

                # All Categorical Features
                if "All Categorical Feature (ClickML Feature)" in fill_feat:
                    fill_feat.remove("All Categorical Feature (ClickML Feature)")
                    fill_feat += df.select_dtypes(include=np.object).columns.tolist()

                
                # Drop Rows
                if strategy == "Drop Rows":
                    st.session_state.all_the_process += f"""
# Drop Rows
df[{fill_feat}] = df[{fill_feat}].dropna(axis=0)
\n """
                    df[fill_feat] = df[fill_feat].dropna(axis=0)
                    st.session_state['df'] = df
                    st.success(f"Missing values have been dropped from the DataFrame for the features **`{fill_feat}`**.")


                # Drop Columns
                elif strategy == "Drop Columns":
                    st.session_state.all_the_process += f"""
# Drop Columns
df[{fill_feat}] = df[{fill_feat}].dropna(axis=1)
\n """
                    df[fill_feat] = df[fill_feat].dropna(axis=1)
                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** have been dropped from the DataFrame.")


                # Fill with Mean
                elif strategy == "Fill with Mean":
                    st.session_state.all_the_process += f"""
# Fill with Mean
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
df[{fill_feat}] = num_imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='mean')
                    df[fill_feat] = num_imputer.fit_transform(df[fill_feat])

                    null_cat = df[missing_df_cols].select_dtypes(include=np.object).columns.tolist()
                    if null_cat:
                        st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
\n """
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        df[null_cat] = cat_imputer.fit_transform(df[null_cat])

                    st.session_state['df'] = df
                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean.")
                    

                # Fill with Median
                elif strategy == "Fill with Median":
                    st.session_state.all_the_process += f"""
# Fill with Median
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
df[{fill_feat}] = pd.DataFrame(num_imputer.fit_transform(df[{fill_feat}]), columns=df[{fill_feat}].columns)
\n """
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='median')
                    df[fill_feat] = pd.DataFrame(num_imputer.fit_transform(df[fill_feat]), columns=df[fill_feat].columns)

                    null_cat = df[missing_df_cols].select_dtypes(include=np.object).columns.tolist()
                    if null_cat:
                        st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
\n """
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        df[null_cat] = cat_imputer.fit_transform(df[null_cat])

                    st.session_state['df'] = df
                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Median. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Median.")


                # Fill with Mode
                elif strategy == "Fill with Mode (Most Frequent)":
                    st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df[{fill_feat}] = imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[fill_feat] = imputer.fit_transform(df[fill_feat])

                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** has been filled with the Mode.")


                # Fill with ffill, bfill
                elif strategy == "Fill with ffill, bfill":
                    st.session_state.all_the_process += f"""
# Fill with ffill, bfill
df[{fill_feat}] = df[{fill_feat}].fillna(method='ffill').fillna(method='bfill')
\n """
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    st.session_state['df'] = df
                    st.success("The DataFrame has been filled with ffill, bfill.")
        
        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="missing_show_df")
        if show_df:
            st.dataframe(df, use_container_width=True)


    # Encoding
    new_line()
    st.markdown("### 🔠 Handling Categorical Data", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Encoding"):
        new_line()

        # Explain
        exp_enc = st.checkbox("Explain Encoding", value=False, key='exp_enc')
        if exp_enc:
            col1, col2 = st.columns([0.8,1])
            with col1:
                st.markdown("<h6 align='center'>Ordinal Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns(2)
                with cola:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a','b','c','b','a']) ),width=120, height=200)
                with colb:
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

            new_line()
        
        # INFO
        show_cat = st.checkbox("Show Categorical Features", value=False, key='show_cat')
        # new_line()
        if show_cat:
            col1, col2 = st.columns(2)
            col1.dataframe(df.select_dtypes(include=np.object), height=250, use_container_width=True )
            if len(df.select_dtypes(include=np.object).columns.tolist()) > 1:
                tmp = df.select_dtypes(include=np.object)
                tmp = tmp.apply(lambda x: x.unique())
                tmp = tmp.to_frame()
                tmp.columns = ['Unique Values']
                col2.dataframe(tmp, height=250, use_container_width=True )
            
        # Further Analysis
        # new_line()
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




        # INPUT
        col1, col2 = st.columns(2)
        with col1:
            enc_feat = st.multiselect("Select Features", df.select_dtypes(include=np.object).columns.tolist(), key='encoding_feat', help="Select the categorical features to encode.")

        with col2:
            encoding = st.selectbox("Select Encoding", ["Select", "Ordinal Encoding", "One Hot Encoding", "Count Frequency Encoding"], key='encoding', help="Select the encoding method.")


        if enc_feat and encoding != "Select":
            new_line()
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Apply", key='encoding_apply',use_container_width=True ,help="Click to apply encoding."):
                progress_bar()
                # Ordinal Encoding
                new_line()
                if encoding == "Ordinal Encoding":
                    st.session_state.all_the_process += f"""
# Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
cat_cols = {enc_feat}
df[cat_cols] = encoder.fit_transform(df[cat_cols])
\n """
                    from sklearn.preprocessing import OrdinalEncoder
                    encoder = OrdinalEncoder()
                    cat_cols = enc_feat
                    df[cat_cols] = encoder.fit_transform(df[cat_cols])
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Ordinal Encoding.")
                    
                # One Hot Encoding
                elif encoding == "One Hot Encoding":
                    st.session_state.all_the_process += f"""
# One Hot Encoding
df = pd.get_dummies(df, columns={enc_feat})
\n """
                    df = pd.get_dummies(df, columns=enc_feat)
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using One Hot Encoding.")

                # Count Frequency Encoding
                elif encoding == "Count Frequency Encoding":
                    st.session_state.all_the_process += f"""
# Count Frequency Encoding
df[{enc_feat}] = df[{enc_feat}].apply(lambda x: x.map(len(df) / x.value_counts()))
\n """
                    df[enc_feat] = df[enc_feat].apply(lambda x: x.map(len(df) / x.value_counts()))
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Count Frequency Encoding.")

        # Show DataFrame Button
        # new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="cat_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)


    # Scaling
    new_line()
    st.markdown("### ⚖️ Scaling", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Scaling"):
        new_line()






        # Scaling Methods
        scaling_methods = st.checkbox("Explain Scaling Methods", value=False, key='scaling_methods')
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

            # write z in the range for the output in latex
            st.latex(r''' **  Z = The\ Scaled\ Value  ** ''')

            new_line()


        # Ranges for the numeric features
        feat_range = st.checkbox("Further Analysis", value=False, key='feat_range')
        if feat_range:
            new_line()
            st.write("The Ranges for the numeric features:")
            col1, col2, col3 = st.columns([0.05,1, 0.05])
            with col2:
                 st.dataframe(df.describe().T, width=700)
            
            new_line()

        # INPUT
        new_line()
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            scale_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features to be scaled.")

        with col2:
            scaling = st.selectbox("Select Scaling", ["Select", "Standard Scaling", "MinMax Scaling", "Robust Scaling"], help="Select the scaling method.")


        if scale_feat and scaling != "Select":       
                new_line()
                col1, col2, col3 = st.columns([1, 0.5, 1])
                
                if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling."):

                    progress_bar()
    
                    # Standard Scaling
                    if scaling == "Standard Scaling":
                        st.session_state.all_the_process += f"""
# Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Standard Scaling.")
    
                    # MinMax Scaling
                    elif scaling == "MinMax Scaling":
                        st.session_state.all_the_process += f"""
# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using MinMax Scaling.")
    
                    # Robust Scaling
                    elif scaling == "Robust Scaling":
                        st.session_state.all_the_process += f"""
# Robust Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Robust Scaling.")

        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="scaling_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)


    # Data Transformation
    new_line()
    st.markdown("### 🧬 Data Transformation", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Transformation"):
        new_line()
        


        # Transformation Methods
        trans_methods = st.checkbox("Explain Transformation Methods", key="trans_methods", value=False)
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

                progress_bar()

                # new_line()
                # Log Transformation
                if trans == "Log Transformation":
                    st.session_state.all_the_process += f"""
#Log Transformation
df[{trans_feat}] = np.log1p(df[{trans_feat}])
\n """
                    df[trans_feat] = np.log1p(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Log Transformation.")

                # Square Root Transformation
                elif trans == "Square Root Transformation":
                    st.session_state.all_the_process += f"""
#Square Root Transformation
df[{trans_feat}] = np.sqrt(df[{trans_feat}])
\n """
                    df[trans_feat] = np.sqrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Square Root Transformation.")

                # Cube Root Transformation
                elif trans == "Cube Root Transformation":
                    st.session_state.all_the_process += f"""
#Cube Root Transformation
df[{trans_feat}] = np.cbrt(df[{trans_feat}])
\n """
                    df[trans_feat] = np.cbrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Cube Root Transformation.")

                # Exponential Transformation
                elif trans == "Exponential Transformation":
                    st.session_state.all_the_process += f"""
#Exponential Transformation
df[{trans_feat}] = np.exp(df[{trans_feat}])
\n """
                    df[trans_feat] = np.exp(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Exponential Transformation.")

        # Show DataFrame Button
        # new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="trans_show_df", help="Click to show the DataFrame.")
        
        if show_df:
            st.dataframe(df, use_container_width=True)


    # Feature Engineering
    new_line()
    st.markdown("### ⚡ Feature Engineering", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Feature Engineering"):

        # Feature Extraction
        new_line()
        st.markdown("#### Feature Extraction", unsafe_allow_html=True)
        new_line()

        col1, col2, col3 = st.columns(3)
        with col1:  
            feat1 = st.selectbox("First Feature/s", ["Select"] + df.select_dtypes(include=np.number).columns.tolist(), key="feat_ex1", help="Select the first feature/s you want to extract.")
        with col2:
            op = st.selectbox("Mathematical Operation", ["Select", "Addition +", "Subtraction -", "Multiplication *", "Division /"], key="feat_ex_op", help="Select the mathematical operation you want to apply.")
        with col3:
            feat2 = st.selectbox("Second Feature/s",["Select"] + df.select_dtypes(include=np.number).columns.tolist(), key="feat_ex2", help="Select the second feature/s you want to extract.")

        if feat1 and op != "Select" and feat2:
            col1, col2, col3 = st.columns(3)
            with col2:
                feat_name = st.text_input("Feature Name", key="feat_name", help="Enter the name of the new feature.")

            col1, col2, col3 = st.columns([1, 0.6, 1])
            new_line()
            if col2.button("Extract Feature"):
                if feat_name == "":
                    feat_name = f"({feat1} {op} {feat2})"

                if op == "Addition +":
                    st.session_state.all_the_process += f"""
# Feature Extraction - Addition
df[{feat_name}] = df[{feat1}] + df[{feat2}]
\n """
                    df[feat_name] = df[feat1] + df[feat2]
                    st.session_state['df'] = df
                    st.success(f"Feature '**_{feat_name}_**' has been extracted using Addition.")

                elif op == "Subtraction -":
                    st.session_state.all_the_process += f"""
# Feature Extraction - Subtraction
df[{feat_name}] = df[{feat1}] - df[{feat2}]
\n """
                    df[feat_name] = df[feat1] - df[feat2]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Subtraction.")

                elif op == "Multiplication *":
                    st.session_state.all_the_process += f"""
# Feature Extraction - Multiplication
df[{feat_name}] = df[{feat1}] * df[{feat2}]
\n """
                    df[feat_name] = df[feat1] * df[feat2]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Multiplication.")

                elif op == "Division /":
                    st.session_state.all_the_process += f"""
# Feature Extraction - Division
df[{feat_name}] = df[{feat1}] / df[{feat2}]
\n """
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
            new_line()
            col1, col2, col3 = st.columns([1, 0.7, 1])
            if col2.button("Transform Feature"):
                if op == "Addition +":
                    st.session_state.all_the_process += f"""
# Feature Transformation - Addition
df[{feat_trans}] = df[{feat_trans}] + {value}
\n """
                    df[feat_trans] = df[feat_trans] + float(value)
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_trans}`** have been transformed using Addition with the value **`{value}`**.")

                elif op == "Subtraction -":
                    st.session_state.all_the_process += f"""
# Feature Transformation - Subtraction
df[{feat_trans}] = df[{feat_trans}] - {value}
\n """
                    df[feat_trans] = df[feat_trans] - float(value)
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_trans}`** have been transformed using Subtraction with the value **`{value}`**.")

                elif op == "Multiplication *":
                    st.session_state.all_the_process += f"""
# Feature Transformation - Multiplication
df[{feat_trans}] = df[{feat_trans}] * {value}
\n """
                    df[feat_trans] = df[feat_trans] * float(value)
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_trans}`** have been transformed using Multiplication with the value **`{value}`**.")

                elif op == "Division /":
                    st.session_state.all_the_process += f"""
# Feature Transformtaion - Division
df[{feat_trans}] = df[{feat_trans}] / {value}
\n """
                    df[feat_trans] = df[feat_trans] / float(value)
                    st.session_state['df'] = df
                    st.success(f"The Featueres **`{feat_trans}`** have been transformed using Division with the value **`{value}`**.")



        # Feature Selection
        st.divider()
        st.markdown("#### Feature Selection", unsafe_allow_html=True)
        new_line()

        feat_sel = st.multiselect("Select Feature/s", df.columns.tolist(), key='feat_sel', help="Select the Features you want to keep in the dataset")
        new_line()

        if feat_sel:
            col1, col2, col3 = st.columns([1, 0.7, 1])
            if col2.button("Select Features"):
                st.session_state.all_the_process += f"""
# Feature Selection\ndf = df[{feat_sel}]
\n """
                progress_bar()
                new_line()
                df = df[feat_sel]
                st.session_state['df'] = df
                st.success(f"The Features **`{feat_sel}`** have been selected.")
        
        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="feat_eng_show_df", help="Click to show the DataFrame.")
        
        if show_df:
            st.dataframe(df, use_container_width=True)


    # Data Splitting
    st.markdown("### 🪚 Data Splitting", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Splitting"):

        new_line()
        train_size, val_size, test_size = 0,0,0
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
            st.session_state['target_variable'] = target
        with col2:
            sets = st.selectbox("Select The Split Sets", ["Select", "Train and Test", "Train, Validation, and Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model. ")
            st.session_state['split_sets'] = sets

        if sets != "Select" and target:
            if sets == "Train, Validation, and Test" :
                new_line()
                col1, col2, col3 = st.columns(3)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    train_size = round(train_size, 2)
                with col2:
                    val_size = st.number_input("Validation Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='val_size')
                    val_size = round(val_size, 2)
                with col3:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='test_size')
                    test_size = round(test_size, 2)

                if float(train_size + val_size + test_size) != 1.0:
                    new_line()
                    st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **validation** + **test** = **{train_size}** + **{val_size}** + **{test_size}** = **{sum([train_size, val_size, test_size])}**" )
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data", use_container_width=True)
                        
                        if split_button:
                            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= {val_size} / (1.0 - {train_size}),random_state=42)
\n """
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


            elif sets == "Train and Test":

                new_line()
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                    train_size = round(train_size, 2)
                with col2:
                    test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='val_size')
                    test_size = round(test_size, 2)

                if float(train_size + test_size) != 1.0:
                    new_line()
                    st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **test** = **{train_size}** + **{test_size}** = **{sum([train_size, test_size])}**" )
                    new_line()

                else:
                    split_button = ""
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col2:
                        new_line()
                        split_button = st.button("Split Data")

                        if split_button:
                            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
\n """
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
                            st.download_button("Download Train Set", train_csv, "train.csv", key='train2')

                        with col2:
                            st.write("Test Set")
                            st.write("X test Shape: ", X_test.shape)
                            st.write("Y test Shape: ", y_test.shape)

                            test = pd.concat([X_test, y_test], axis=1)
                            test_csv = test.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Test Set", test_csv, "test.csv", key='test2')


    # Building the model
    new_line()
    st.markdown("### 🤖 Building the Model")
    new_line()
    problem_type = ""
    with st.expander(" Model Building"):    
        
        target, problem_type, model = "", "", ""
        col1, col2, col3 = st.columns(3)

        with col1:
            target = st.selectbox("Target Variable", [st.session_state['target_variable']] , key='target_ml', help="The target variable is the variable that you want to predict")
            new_line()

        with col2:
            problem_type = st.selectbox("Problem Type", ["Select", "Classification", "Regression"], key='problem_type', help="The problem type is the type of problem that you want to solve")

        with col3:

            if problem_type == "Classification":
                model = st.selectbox("Model", ["Select", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()

            elif problem_type == "Regression":
                model = st.selectbox("Model", ["Linear Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()


        if target != "Select" and problem_type and model:
            
            if problem_type == "Classification":
                 
                if model == "Logistic Regression":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        penalty = st.selectbox("Penalty (Optional)", ["l2", "l1", "none", "elasticnet"], key='penalty')

                    with col2:
                        solver = st.selectbox("Solver (Optional)", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"], key='solver')

                    with col3:
                        C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')

                    
                    col1, col2, col3 = st.columns([1,1,1])
                    if col2.button("Train Model", use_container_width=True):
                        
                        
                        progress_bar()

                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='{penalty}', solver='{solver}', C={C}, random_state=42)
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True,  key='save_model')

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors **Required**", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')

                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')

                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')

                    
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):
                        progress_bar()

                        st.session_state['trained_model_bool'] = True

                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Support Vector Machine":
                        
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["rbf", "poly", "linear", "sigmoid", "precomputed"], key='kernel')
    
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=100, value=3, step=1, key='degree')
    
                    with col3:
                        C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')
    
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):

                        progress_bar()
                        st.session_state['trained_model_bool'] = True
    
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Support Vector Machine
from sklearn.svm import SVC
model = SVC(kernel='{kernel}', degree={degree}, C={C}, random_state=42)
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Decision Tree":
                            
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
        
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
        
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                            
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
        
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split}, random_state=42)
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "Random Forest":
                                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                                
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", use_container_width=True):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split}, random_state=42)
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}', random_state=42)
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> LightGBM
from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)
model.fit(X_train, y_train)
\n """
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> CatBoost
from catboost import CatBoostClassifier
model = CatBoostClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')      

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept={fit_intercept}, positive={positive}, copy_X={copy_x})
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Support Vector Machine
from sklearn.svm import SVR
model = SVR(kernel='{kernel}', degree={degree}, gamma='{gamma}')
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split})
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')
                
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split})
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                if model == "XGBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0001, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model"):
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}')
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> LightGBM
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model') 

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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> CatBoost
from catboost import CatBoostRegressor
model = CatBoostRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')
model.fit(X_train, y_train)
\n """
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
                        col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')


    # Evaluation
    if st.session_state['trained_model_bool']:
        st.markdown("### 📈 Evaluation")
        new_line()
        with st.expander("Model Evaluation"):
            # Load the model
            import joblib
            model = joblib.load('model.pkl')
            

            if str(model) not in st.session_state.lst_models_predctions:
                
                st.session_state.lst_models_predctions.append(str(model))
                st.session_state.lst_models.append(str(model))
                if str(model) not in st.session_state.models_with_eval.keys():
                    st.session_state.models_with_eval[str(model)] = []


                

                # Predictions
                if st.session_state["split_sets"] == "Train, Validation, and Test":
                        
                        st.session_state.all_the_process += f"""
# Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)
\n """
                        y_pred_train = model.predict(X_train)
                        st.session_state.y_pred_train = y_pred_train
                        y_pred_val = model.predict(X_val)
                        st.session_state.y_pred_val = y_pred_val
                        y_pred_test = model.predict(X_test)
                        st.session_state.y_pred_test = y_pred_test


                elif st.session_state["split_sets"] == "Train and Test":
                    
                    st.session_state.all_the_process += f"""
# Predictions 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
\n """  
                    
                    y_pred_train = model.predict(X_train)
                    st.session_state.y_pred_train = y_pred_train
                    y_pred_test = model.predict(X_test)
                    st.session_state.y_pred_test = y_pred_test

            # Choose Evaluation Metric
            if st.session_state['problem_type'] == "Classification":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"], key='evaluation_metric')

            elif st.session_state['problem_type'] == "Regression":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2 Score"], key='evaluation_metric')

            
            col1, col2, col3 = st.columns([1, 0.6, 1])
            
            st.session_state.show_eval = True
                
            
            if evaluation_metric != []:
                

                for metric in evaluation_metric:


                        if metric == "Accuracy":

                            # Check if Accuary is element of the list of that model
                            if "Accuracy" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Accuracy")

                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - Accuracy 
from sklearn.metrics import accuracy_score
print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
print("Accuracy Score on Validation Set: ", accuracy_score(y_val, y_pred_val))
print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import accuracy_score
                                    train_acc = accuracy_score(y_train, y_pred_train)
                                    val_acc = accuracy_score(y_val, y_pred_val)
                                    test_acc = accuracy_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_acc, val_acc, test_acc]
                                    st.session_state['metrics_df'] = metrics_df


                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))
\n """

                                    from sklearn.metrics import accuracy_score
                                    train_acc = accuracy_score(y_train, y_pred_train)
                                    test_acc = accuracy_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_acc, test_acc]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Precision":
                            
                            if "Precision" not in st.session_state.models_with_eval[str(model)]:
                                
                                st.session_state.models_with_eval[str(model)].append("Precision")

                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - Precision
from sklearn.metrics import precision_score
print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
print("Precision Score on Validation Set: ", precision_score(y_val, y_pred_val))
print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import precision_score
                                    train_prec = precision_score(y_train, y_pred_train)
                                    val_prec = precision_score(y_val, y_pred_val)
                                    test_prec = precision_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_prec, val_prec, test_prec]
                                    st.session_state['metrics_df'] = metrics_df
                                    
                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - Precision
from sklearn.metrics import precision_score
print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import precision_score
                                    train_prec = precision_score(y_train, y_pred_train)
                                    test_prec = precision_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_prec, test_prec]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Recall":

                            if "Recall" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Recall")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - Recall
from sklearn.metrics import recall_score
print("Recall Score on Train Set: ", recall_score(y_train, y_pred_train))
print("Recall Score on Validation Set: ", recall_score(y_val, y_pred_val))
print("Recall Score on Test Set: ", recall_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import recall_score
                                    train_rec = recall_score(y_train, y_pred_train)
                                    val_rec = recall_score(y_val, y_pred_val)
                                    test_rec = recall_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_rec, val_rec, test_rec]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - Recall
from sklearn.metrics import recall_score
print("Recall Score on Train Set: ", recall_score(y_train, y_pred_train))
print("Recall Score on Test Set: ", recall_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import recall_score
                                    train_rec = recall_score(y_train, y_pred_train)
                                    test_rec = recall_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_rec, test_rec]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "F1 Score":

                            if "F1 Score" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("F1 Score")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - F1 Score
from sklearn.metrics import f1_score
print("F1 Score on Train Set: ", f1_score(y_train, y_pred_train))
print("F1 Score on Validation Set: ", f1_score(y_val, y_pred_val))
print("F1 Score on Test Set: ", f1_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import f1_score
                                    train_f1 = f1_score(y_train, y_pred_train)
                                    val_f1 = f1_score(y_val, y_pred_val)
                                    test_f1 = f1_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_f1, val_f1, test_f1]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - F1 Score
from sklearn.metrics import f1_score
print("F1 Score on Train Set: ", f1_score(y_train, y_pred_train))
print("F1 Score on Test Set: ", f1_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import f1_score
                                    train_f1 = f1_score(y_train, y_pred_train)
                                    test_f1 = f1_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_f1, test_f1]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "AUC Score":

                            if "AUC Score" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("AUC Score")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - AUC Score
from sklearn.metrics import roc_auc_score
print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
print("AUC Score on Validation Set: ", roc_auc_score(y_val, y_pred_val))
print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import roc_auc_score
                                    train_auc = roc_auc_score(y_train, y_pred_train)
                                    val_auc = roc_auc_score(y_val, y_pred_val)
                                    test_auc = roc_auc_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_auc, val_auc, test_auc]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - AUC Score
from sklearn.metrics import roc_auc_score
print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import roc_auc_score
                                    train_auc = roc_auc_score(y_train, y_pred_train)
                                    test_auc = roc_auc_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_auc, test_auc]
                                    st.session_state['metrics_df'] = metrics_df
                            

                        elif metric == "Mean Absolute Error (MAE)":

                            if "Mean Absolute Error (MAE)" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Mean Absolute Error (MAE)")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - MAE
from sklearn.metrics import mean_absolute_error
print("MAE on Train Set: ", mean_absolute_error(y_train, y_pred_train))
print("MAE on Validation Set: ", mean_absolute_error(y_val, y_pred_val))
print("MAE on Test Set: ", mean_absolute_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_absolute_error
                                    train_mae = mean_absolute_error(y_train, y_pred_train)
                                    val_mae = mean_absolute_error(y_val, y_pred_val)
                                    test_mae = mean_absolute_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mae, val_mae, test_mae]
                                    st.session_state['metrics_df'] = metrics_df

                                else:
                                    st.session_state.all_the_process += f"""
# Evaluation - MAE
from sklearn.metrics import mean_absolute_error
print("MAE on Train Set: ", mean_absolute_error(y_train, y_pred_train))
print("MAE on Test Set: ", mean_absolute_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_absolute_error
                                    train_mae = mean_absolute_error(y_train, y_pred_train)
                                    test_mae = mean_absolute_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mae, test_mae]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Mean Squared Error (MSE)":

                            if "Mean Squared Error (MSE)" not in st.session_state.models_with_eval[str(model)]:
                                
                                st.session_state.models_with_eval[str(model)].append("Mean Squared Error (MSE)")

                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - MSE
from sklearn.metrics import mean_squared_error
print("MSE on Train Set: ", mean_squared_error(y_train, y_pred_train))
print("MSE on Validation Set: ", mean_squared_error(y_val, y_pred_val))
print("MSE on Test Set: ", mean_squared_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_mse = mean_squared_error(y_train, y_pred_train)
                                    val_mse = mean_squared_error(y_val, y_pred_val)
                                    test_mse = mean_squared_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mse, val_mse, test_mse]
                                    st.session_state['metrics_df'] = metrics_df

                                else:

                                    st.session_state.all_the_process += f"""
# Evaluation - MSE
from sklearn.metrics import mean_squared_error
print("MSE on Train Set: ", mean_squared_error(y_train, y_pred_train))
print("MSE on Test Set: ", mean_squared_error(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_mse = mean_squared_error(y_train, y_pred_train)
                                    test_mse = mean_squared_error(y_test, y_pred_test)

                                    metrics_df[metric] = [train_mse, test_mse]
                                    st.session_state['metrics_df'] = metrics_df


                        elif metric == "Root Mean Squared Error (RMSE)":

                            if "Root Mean Squared Error (RMSE)" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("Root Mean Squared Error (RMSE)")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - RMSE
from sklearn.metrics import mean_squared_error
print("RMSE on Train Set: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("RMSE on Validation Set: ", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("RMSE on Test Set: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                    metrics_df[metric] = [train_rmse, val_rmse, test_rmse]
                                    st.session_state['metrics_df'] = metrics_df

                                else:

                                    st.session_state.all_the_process += f"""
# Evaluation - RMSE
from sklearn.metrics import mean_squared_error
print("RMSE on Train Set: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("RMSE on Test Set: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
\n """
                                    from sklearn.metrics import mean_squared_error
                                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                    metrics_df[metric] = [train_rmse, test_rmse]
                                    st.session_state['metrics_df'] = metrics_df

                            
                        elif metric == "R2 Score":

                            if "R2 Score" not in st.session_state.models_with_eval[str(model)]:

                                st.session_state.models_with_eval[str(model)].append("R2 Score")
                            
                                if st.session_state["split_sets"] == "Train, Validation, and Test":

                                    st.session_state.all_the_process += f"""
# Evaluation - R2 Score
from sklearn.metrics import r2_score
print("R2 Score on Train Set: ", r2_score(y_train, y_pred_train))
print("R2 Score on Validation Set: ", r2_score(y_val, y_pred_val))
print("R2 Score on Test Set: ", r2_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import r2_score
                                    train_r2 = r2_score(y_train, y_pred_train)
                                    val_r2 = r2_score(y_val, y_pred_val)
                                    test_r2 = r2_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_r2, val_r2, test_r2]
                                    st.session_state['metrics_df'] = metrics_df

                                else:

                                    st.session_state.all_the_process += f"""
# Evaluation - R2 Score
from sklearn.metrics import r2_score
print("R2 Score on Train Set: ", r2_score(y_train, y_pred_train))
print("R2 Score on Test Set: ", r2_score(y_test, y_pred_test))
\n """
                                    from sklearn.metrics import r2_score
                                    train_r2 = r2_score(y_train, y_pred_train)
                                    test_r2 = r2_score(y_test, y_pred_test)

                                    metrics_df[metric] = [train_r2, test_r2]
                                    st.session_state['metrics_df'] = metrics_df



                # Show Evaluation Metric
                if show_eval:
                    new_line()
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    st.markdown("### Evaluation Metric")

                    if st.session_state["split_sets"] == "Train, Validation, and Test":
                        st.session_state['metrics_df'].index = ['Train', 'Validation', 'Test']
                        st.write(st.session_state['metrics_df'])

                    elif st.session_state["split_sets"] == "Train and Test":
                        st.session_state['metrics_df'].index = ['Train', 'Test']
                        st.write(st.session_state['metrics_df'])

                    


                    # Show Evaluation Metric Plot
                    new_line()
                    st.markdown("### Evaluation Metric Plot")
                    st.line_chart(st.session_state['metrics_df'])

                    # Show ROC Curve as plot
                    if "AUC Score" in evaluation_metric:
                        from sklearn.metrics import plot_roc_curve
                        st.markdown("### ROC Curve")
                        new_line()
                        
                        if st.session_state["split_sets"] == "Train, Validation, and Test":

                            # Show the ROC curve plot without any columns
                            col1, col2, col3 = st.columns([0.2, 1, 0.2])
                            fig, ax = plt.subplots()
                            plot_roc_curve(model, X_train, y_train, ax=ax)
                            plot_roc_curve(model, X_val, y_val, ax=ax)
                            plot_roc_curve(model, X_test, y_test, ax=ax)
                            ax.legend(['Train', 'Validation', 'Test'])
                            col2.pyplot(fig, legend=True)

                        elif st.session_state["split_sets"] == "Train and Test":

                            # Show the ROC curve plot without any columns
                            col1, col2, col3 = st.columns([0.2, 1, 0.2])
                            fig, ax = plt.subplots()
                            plot_roc_curve(model, X_train, y_train, ax=ax)
                            plot_roc_curve(model, X_test, y_test, ax=ax)
                            ax.legend(['Train', 'Test'])
                            col2.pyplot(fig, legend=True)

                            

                    # Show Confusion Matrix as plot
                    if st.session_state['problem_type'] == "Classification":
                        # from sklearn.metrics import plot_confusion_matrix
                        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                        st.markdown("### Confusion Matrix")
                        new_line()

                        cm = confusion_matrix(y_test, y_pred_test)
                        col1, col2, col3 = st.columns([0.2,1,0.2])
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax)
                        col2.pyplot(fig)
                        
                        # Show the confusion matrix plot without any columns
                        # col1, col2, col3 = st.columns([0.2, 1, 0.2])
                        # fig, ax = plt.subplots()
                        # plot_confusion_matrix(model, X_test, y_test, ax=ax)
                        # col2.pyplot(fig)

                     
    st.divider()          
    col1, col2, col3, col4= st.columns(4, gap='small')        

    if col1.button("🎬 Show df", use_container_width=True):
        new_line()
        st.subheader(" 🎬 Show The Dataframe")
        st.write("The dataframe is the dataframe that is used on this application to build the Machine Learning model. You can see the dataframe below 👇")
        new_line()
        st.dataframe(df, use_container_width=True)

    st.session_state.df.to_csv("df.csv", index=False)
    df_file = open("df.csv", "rb")
    df_bytes = df_file.read()
    if col2.download_button("📌 Download df", df_bytes, "df.csv", key='save_df', use_container_width=True):
        st.success("Downloaded Successfully!")

    if col3.button("💻  Code", use_container_width=True):
        new_line()
        st.subheader("💻  The Code")
        st.write("The code below is the code that is used to build the model. It is the code that is generated by the app. You can copy the code and use it in your own project 😉")
        new_line()
        st.code(st.session_state.all_the_process, language='python')

    if col4.button("⛔ Reset", use_container_width=True):
        new_line()
        st.subheader("⛔ Reset")
        st.write("Click the button below to reset the app and start over again")
        new_line()
        st.session_state.reset_1 = True

    if st.session_state.reset_1:
        col1, col2, col3 = st.columns(3)
        if col2.button("⛔ Reset", use_container_width=True, key='reset'):
            st.session_state.df = None
            st.session_state.clear()
            st.experimental_rerun()
            
