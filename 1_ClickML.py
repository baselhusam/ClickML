import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

from streamlit_option_menu import option_menu

# Config
st.set_page_config(layout="centered", page_title="Click ML", page_icon="👆")

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


y_pred_train, y_pred_test, y_pred_val = st.session_state.y_pred_train, st.session_state.y_pred_test, st.session_state.y_pred_val
metrics_df = pd.DataFrame()

def new_line():
    st.write("\n")

st.cache_data()
def load_data(upd_file):
    df = pd.read_csv(upd_file)
    return df

def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)
# Title
col1, col2, col3 = st.columns([0.5,1,0.5])
col2.markdown("<h1 align='center'> 👉 ClickML", unsafe_allow_html=True)
# new_line()
# new_line()
# new_line()
# new_line()


# Make Animation Elements using HTML and CSS
css = """
/* 3D tower loader made by: csozi | Website: www.csozi.hu*/

.loader {
  scale: 3;
  height: 50px;
  width: 40px;
}

.box {
  position: relative;
  opacity: 0;
  left: 10px;
}

.side-left {
  position: absolute;
  background-color: #286cb5;
  width: 19px;
  height: 5px;
  transform: skew(0deg, -25deg);
  top: 14px;
  left: 10px;
}

.side-right {
  position: absolute;
  background-color: #2f85e0;
  width: 19px;
  height: 5px;
  transform: skew(0deg, 25deg);
  top: 14px;
  left: -9px;
}

.side-top {
  position: absolute;
  background-color: #5fa8f5;
  width: 20px;
  height: 20px;
  rotate: 45deg;
  transform: skew(-20deg, -20deg);
}

.box-1 {
  animation: from-left 4s infinite;
}

.box-2 {
  animation: from-right 4s infinite;
  animation-delay: 1s;
}

.box-3 {
  animation: from-left 4s infinite;
  animation-delay: 2s;
}

.box-4 {
  animation: from-right 4s infinite;
  animation-delay: 3s;
}

@keyframes from-left {
  0% {
    z-index: 20;
    opacity: 0;
    translate: -20px -6px;
  }

  20% {
    z-index: 10;
    opacity: 1;
    translate: 0px 0px;
  }

  40% {
    z-index: 9;
    translate: 0px 4px;
  }

  60% {
    z-index: 8;
    translate: 0px 8px;
  }

  80% {
    z-index: 7;
    opacity: 1;
    translate: 0px 12px;
  }

  100% {
    z-index: 5;
    translate: 0px 30px;
    opacity: 0;
  }
}

@keyframes from-right {
  0% {
    z-index: 20;
    opacity: 0;
    translate: 20px -6px;
  }

  20% {
    z-index: 10;
    opacity: 1;
    translate: 0px 0px;
  }

  40% {
    z-index: 9;
    translate: 0px 4px;
  }

  60% {
    z-index: 8;
    translate: 0px 8px;
  }

  80% {
    z-index: 7;
    opacity: 1;
    translate: 0px 12px;
  }

  100% {
    z-index: 5;
    translate: 0px 30px;
    opacity: 0;
  }
}
"""
html =  """<div class="loader">
  <div class="box box-1">
    <div class="side-left"></div>
    <div class="side-right"></div>
    <div class="side-top"></div>
  </div>
  <div class="box box-2">
    <div class="side-left"></div>
    <div class="side-right"></div>
    <div class="side-top"></div>
  </div>
  <div class="box box-3">
    <div class="side-left"></div>
    <div class="side-right"></div>
    <div class="side-top"></div>
  </div>
  <div class="box box-4">
    <div class="side-left"></div>
    <div class="side-right"></div>
    <div class="side-top"></div>
  </div>
</div>"""

pm_css = """
.container {
  position: absolute;
  top: 45%;
  left: 48%;
}

.square {
  width: 8px;
  height: 30px;
  background: rgb(71, 195, 248);
  border-radius: 10px;
  display: block;
  /*margin:10px;*/
  -webkit-animation: turn 2.5s ease infinite;
  animation: turn 2.5s ease infinite;
  box-shadow: rgb(71, 195, 248) 0px 1px 15px 0px;
}

.top {
  position: absolute;
  left: 40%;
  top: 50%;
  -webkit-transform: rotate(90deg);
  transform: rotate(90deg);
}

.bottom {
  position: absolute;
  left: 40%;
  top: 50%;
  -webkit-transform: rotate(-90deg);
  transform: rotate(-90deg);
}

.left {
  position: absolute;
  left: 40%;
  top: 50%;
}

.right {
  position: absolute;
  left: 40%;
  top: 50%;
  -webkit-transform: rotate(-180deg);
  transform: rotate(-180deg);
}

@-webkit-keyframes turn {
  0% {
    transform: translateX(0) translateY(0) rotate(0);
  }

  50% {
    transform: translateX(400%) translateY(100%) rotate(90deg);
  }

  100% {
    transform: translateX(0) translateY(0) rotate(0);
  }
}

@keyframes turn {
  0% {
    transform: translateX(0) translateY(0) rotate(0);
  }

  70% {
    transform: translateX(400%) translateY(100%) rotate(90deg);
  }

  100% {
    transform: translateX(0) translateY(0) rotate(0);
  }
}"""
pm_html = """
<div class="container">
  <div class="top">
    <div class="square">
      <div class="square">
        <div class="square">
          <div class="square">
            <div class="square"><div class="square">
            </div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="bottom">
    <div class="square">
      <div class="square">
        <div class="square">
          <div class="square">
            <div class="square"><div class="square">
            </div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="left">
    <div class="square">
      <div class="square">
        <div class="square">
          <div class="square">
            <div class="square"><div class="square">
            </div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="square">
      <div class="square">
        <div class="square">
          <div class="square">
            <div class="square"><div class="square">
            </div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>""" 
# col1, col2, col3 = st.columns([1,0.01,1])
# with col2:
#     st.markdown("<style>" + pm_css + "</style>", unsafe_allow_html=True)
#     st.markdown(pm_html, unsafe_allow_html=True)


# new_line()
# new_line()
# new_line()
# new_line()
# new_line()

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
    # st.image("./logo2.png",  use_column_width=True)
    new_line()
    new_line()
    new_line()
    new_line()
    
    bar_css = """
    .loader {
    position: relative;
    width: 120px;
    height: 90px;
    margin: 0 auto;
    }

    .loader:before {
    content: "";
    position: absolute;
    bottom: 30px;
    left: 50px;
    height: 30px;
    width: 30px;
    border-radius: 50%;
    background: #2a9d8f;
    animation: loading-bounce 0.5s ease-in-out infinite alternate;
    }

    .loader:after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    height: 7px;
    width: 45px;
    border-radius: 4px;
    box-shadow: 0 5px 0 #d09f5e, -35px 50px 0 #d09f5e, -70px 95px 0 #d09f5e;
    animation: loading-step 1s ease-in-out infinite;
    }

    @keyframes loading-bounce {
    0% {
        transform: scale(1, 0.7);
    }

    40% {
        transform: scale(0.8, 1.2);
    }

    60% {
        transform: scale(1, 1);
    }

    100% {
        bottom: 140px;
    }
    }

    @keyframes loading-step {
    0% {
        box-shadow: 0 10px 0 rgba(0, 0, 0, 0),
                0 10px 0 #d09f5e,
                -35px 50px 0 #d09f5e,
                -70px 90px 0 #d09f5e;
    }

    100% {
        box-shadow: 0 10px 0 #d09f5e,
                -35px 50px 0 #d09f5e,
                -70px 90px 0 #d09f5e,
                -70px 90px 0 rgba(0, 0, 0, 0);
    }
    }
    """
    bar_html = """
    <div class="loader"></div>
    """

    st.markdown("<style>" + bar_css + "</style>", unsafe_allow_html=True)
    st.markdown(bar_html, unsafe_allow_html=True)
    
    

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

        # Correlation Matrix using heatmap seabron
        corr = st.checkbox("Show Correlation", value=False)
        new_line()
        if corr:
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), cmap='Blues', annot=True, ax=ax)
            st.pyplot(fig)

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
                st.session_state.all_the_process += f"# Delete Columns\ndf.drop(columns={col_to_delete}, inplace=True)\n\n"
                progress_bar()
                new_line()
                df.drop(columns=col_to_delete, inplace=True)
                st.session_state.df = df
                st.success(f"The Columns **`{col_to_delete}`** are Deleted Successfully!")


        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame")
        # new_line()

        if show_df:
            st.dataframe(df, use_container_width=True)
                


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
            missing_df_cols = df.columns[df.isnull().any()].tolist()
            fill_feat = st.multiselect("Select Features", missing_df_cols, help="Select Features to fill missing values")

        with col2:
            strategy = st.selectbox("Select Missing Values Strategy", ["Select", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode (Most Frequent)", "Fill with ffill, bfill"], help="Select Missing Values Strategy")


        if fill_feat and strategy != "Select":

            new_line()
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Apply", key="missing_apply", help="Apply Missing Values Strategy"):

                progress_bar()
                
                # Drop Rows
                if strategy == "Drop Rows":
                    st.session_state.all_the_process += f"# Drop Rows\ndf[{fill_feat}] = df[{fill_feat}].dropna(axis=0)\n\n]"
                    df[fill_feat] = df[fill_feat].dropna(axis=0)
                    st.session_state['df'] = df
                    st.success(f"Missing values have been dropped from the DataFrame for the features **`{fill_feat}`**.")


                # Drop Columns
                elif strategy == "Drop Columns":
                    st.session_state.all_the_process += f"# Drop Columns\ndf[{fill_feat}] = df[{fill_feat}].dropna(axis=1)\n\n]"
                    df[fill_feat] = df[fill_feat].dropna(axis=1)
                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** have been dropped from the DataFrame.")


                # Fill with Mean
                elif strategy == "Fill with Mean":
                    st.session_state.all_the_process += f"# Fill with Mean\nfrom sklearn.impute import SimpleImputer\nnum_imputer = SimpleImputer(strategy='mean')\ndf[{fill_feat}] = num_imputer.fit_transform(df[{fill_feat}])\n\n"
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='mean')
                    df[fill_feat] = num_imputer.fit_transform(df[fill_feat])

                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.session_state.all_the_process += f"# Fill with Mode\nfrom sklearn.impute import SimpleImputer\ncat_imputer = SimpleImputer(strategy='most_frequent')\ncat_feat = df.select_dtypes(include=np.object).columns.tolist()\ndf[cat_feat] = cat_imputer.fit_transform(df[cat_feat])\n\n"
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        cat_feat = df.select_dtypes(include=np.object).columns.tolist()
                        df[cat_feat] = cat_imputer.fit_transform(df[cat_feat])

                    st.session_state['df'] = df
                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean. And the categorical columns **`{cat_feat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean.")
                    

                # Fill with Median
                elif strategy == "Fill with Median":
                    st.session_state.all_the_process += f"# Fill with Median\nfrom sklearn.impute import SimpleImputer\nnum_imputer = SimpleImputer(strategy='median')\ndf[{fill_feat}] = pd.DataFrame(num_imputer.fit_transform(df[{fill_feat}]), columns=df[{fill_feat}].columns)\n\n"
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='median')
                    df[fill_feat] = pd.DataFrame(num_imputer.fit_transform(df[fill_feat]), columns=df[fill_feat].columns)

                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.session_state.all_the_process += f"# Fill with Mode\nfrom sklearn.impute import SimpleImputer\ncat_imputer = SimpleImputer(strategy='most_frequent')\ncat_feat = df.select_dtypes(include=np.object).columns.tolist()\ndf[cat_feat] = cat_imputer.fit_transform(df[cat_feat])\n\n"
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        cat_feat = df.select_dtypes(include=np.object).columns.tolist()
                        df[cat_feat] = cat_imputer.fit_transform(df[cat_feat])

                    st.session_state['df'] = df
                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Median. And the categorical columns **`{cat_feat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Median.")


                # Fill with Mode
                elif strategy == "Fill with Mode (Most Frequent)":
                    st.session_state.all_the_process += f"# Fill with Mode\nfrom sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='most_frequent')\ndf[{fill_feat}] = imputer.fit_transform(df[{fill_feat}])\n\n"
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[fill_feat] = imputer.fit_transform(df[fill_feat])

                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** has been filled with the Mode.")


                # Fill with ffill, bfill
                elif strategy == "Fill with ffill, bfill":
                    st.session_state.all_the_process += f"# Fill with ffill, bfill\ndf[{fill_feat}] = df[{fill_feat}].fillna(method='ffill').fillna(method='bfill')\n\n"
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


        # INPUT
        # new_line()
        col1, col2 = st.columns(2)
        with col1:
            enc_feat = st.multiselect("Select Features", df.select_dtypes(include=np.object).columns.tolist(), key='encoding_feat', help="Select the categorical features to encode.")

        with col2:
            encoding = st.selectbox("Select Encoding", ["Select", "Ordinal Encoding", "One Hot Encoding", "Count Frequency Encoding"], key='encoding', help="Select the encoding method.")


        if enc_feat and encoding != "Select":
            col1, col2, col3 = st.columns([1,0.5,1])
            # new_line()
            if col2.button("Apply", key='encoding_apply',use_container_width=True ,help="Click to apply encoding."):
                progress_bar()
                # Ordinal Encoding
                new_line()
                if encoding == "Ordinal Encoding":
                    st.session_state.all_the_process += f"# Ordinal Encoding\nfrom sklearn.preprocessing import OrdinalEncoder\nencoder = OrdinalEncoder()\ncat_cols = {enc_feat}\ndf[cat_cols] = encoder.fit_transform(df[cat_cols])\n\n"
                    from sklearn.preprocessing import OrdinalEncoder
                    encoder = OrdinalEncoder()
                    cat_cols = enc_feat
                    df[cat_cols] = encoder.fit_transform(df[cat_cols])
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Ordinal Encoding.")
                    
                # One Hot Encoding
                elif encoding == "One Hot Encoding":
                    st.session_state.all_the_process += f"# One Hot Encoding\ndf = pd.get_dummies(df, columns={enc_feat})\n\n"
                    df = pd.get_dummies(df, columns=enc_feat)
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using One Hot Encoding.")

                # Count Frequency Encoding
                elif encoding == "Count Frequency Encoding":
                    st.session_state.all_the_process += f"# Count Frequency Encoding\ndf[{enc_feat}] = df[{enc_feat}].apply(lambda x: x.map(len(df) / x.value_counts()))\n\n"
                    df[enc_feat] = df[enc_feat].apply(lambda x: x.map(len(df) / x.value_counts()))
                    st.session_state['df'] = df
                    st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Count Frequency Encoding.")

        # Show DataFrame Button
        # new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="cat_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)


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
                new_line()
                col1, col2, col3 = st.columns([1, 0.5, 1])
                
                if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling."):

                    progress_bar()
    
                    # Standard Scaling
                    if scaling == "Standard Scaling":
                        st.session_state.all_the_process += f"# Standard Scaling\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndf[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)\n\n"
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Standard Scaling.")
    
                    # MinMax Scaling
                    elif scaling == "MinMax Scaling":
                        st.session_state.all_the_process += f"# MinMax Scaling\nfrom sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\ndf[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)\n\n"
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using MinMax Scaling.")
    
                    # Robust Scaling
                    elif scaling == "Robust Scaling":
                        st.session_state.all_the_process += f"# Robust Scaling\nfrom sklearn.preprocessing import RobustScaler\nscaler = RobustScaler()\ndf[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)\n\n"
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

                progress_bar()

                # new_line()
                # Log Transformation
                if trans == "Log Transformation":
                    st.session_state.all_the_process += f"#Log Transformation\ndf[{trans_feat}] = np.log1p(df[{trans_feat}])"
                    df[trans_feat] = np.log1p(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Log Transformation.")

                # Square Root Transformation
                elif trans == "Square Root Transformation":
                    st.session_state.all_the_process += f"#Square Root Transformation\ndf[{trans_feat}] = np.sqrt(df[{trans_feat}])"
                    df[trans_feat] = np.sqrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Square Root Transformation.")

                # Cube Root Transformation
                elif trans == "Cube Root Transformation":
                    st.session_state.all_the_process += f"#Cube Root Transformation\ndf[{trans_feat}] = np.cbrt(df[{trans_feat}])"
                    df[trans_feat] = np.cbrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Cube Root Transformation.")

                # Exponential Transformation
                elif trans == "Exponential Transformation":
                    st.session_state.all_the_process += f"#Exponential Transformation\ndf[{trans_feat} = np.exp(df[{trans_feat}])]"
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
    st.markdown("### Feature Engineering", unsafe_allow_html=True)
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
                    st.session_state.all_the_process += f"# Feature Extraction - Addition\ndf[{feat_name}] = df[{feat1}] + df[{feat2}]\n\n"
                    df[feat_name] = df[feat1] + df[feat2]
                    st.session_state['df'] = df
                    st.success(f"Feature '**_{feat_name}_**' has been extracted using Addition.")

                elif op == "Subtraction -":
                    st.session_state.all_the_process += f"# Feature Extraction - Subtraction\ndf[{feat_name}] = df[{feat1}] - df[{feat2}]\n\n"
                    df[feat_name] = df[feat1] - df[feat2]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Subtraction.")

                elif op == "Multiplication *":
                    st.session_state.all_the_process += f"# Feature Extraction - Multiplication\ndf[{feat_name}] = df[{feat1}] * df[{feat2}]\n\n"
                    df[feat_name] = df[feat1] * df[feat2]
                    st.session_state['df'] = df
                    st.success(f"Feature {feat_name} has been extracted using Multiplication.")

                elif op == "Division /":
                    st.session_state.all_the_process += f"# Feature Extraction - Division\ndf[{feat_name}] = df[{feat1}] / df[{feat2}]\n\n"
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
                    st.session_state.all_the_process += f"# Feature Transformation - Addition\ndf[{feat_trans}] = df[{feat_trans}] + {value}\n\n"
                    df[feat_trans] = df[feat_trans] + float(value)
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_trans}`** have been transformed using Addition with the value **`{value}`**.")

                elif op == "Subtraction -":
                    st.session_state.all_the_process += f"# Feature Transformation - Subtraction\ndf[{feat_trans}] = df[{feat_trans}] - {value}\n\n"
                    df[feat_trans] = df[feat_trans] - float(value)
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_trans}`** have been transformed using Subtraction with the value **`{value}`**.")

                elif op == "Multiplication *":
                    st.session_state.all_the_process += f"# Feature Transformation - Multiplication\ndf[{feat_trans}] = df[{feat_trans}] * {value}\n\n"
                    df[feat_trans] = df[feat_trans] * float(value)
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_trans}`** have been transformed using Multiplication with the value **`{value}`**.")

                elif op == "Division /":
                    st.session_state.all_the_process += f"# Feature Transformtaion - Division\ndf[{feat_trans}] = df[{feat_trans}] / {value}\n\n"
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
                st.session_state.all_the_process += f"# Feature Selection\ndf = df[{feat_sel}]\n\n"
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
                            st.session_state.all_the_process += f"# Data Splitting\nfrom sklearn.model_selection import train_test_split\nX_train, X_rem, y_train, y_rem = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)\nX_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= {val_size} / (1.0 - {train_size}),random_state=42)\n\n"
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
                            st.session_state.all_the_process += f"# Data Splitting\nfrom sklearn.model_selection import train_test_split\nX_train, X_val, y_train, y_val = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)\n\n"
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
                        
                        
                        progress_bar()

                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(penalty='{penalty}', solver='{solver}', C={C}, random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()

                        st.session_state['trained_model_bool'] = True

                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.neighbors import KNeighborsClassifier\nmodel = KNeighborsClassifier(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')\nmodel.fit(X_train, y_train)\n\n"
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

                        progress_bar()
                        st.session_state['trained_model_bool'] = True
    
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.svm import SVC\nmodel = SVC(kernel='{kernel}', degree={degree}, C={C}, random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
        
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split}, random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split}, random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom xgboost import XGBClassifier\nmodel = XGBClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}', random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom lightgbm import LGBMClassifier\nmodel = LGBMClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        st.session_state.all_the_process += f"# Model Building\nfrom catboost import CatBoostClassifier\nmodel = CatBoostClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.linear_model import LinearRegression\nmodel = LinearRegression(fit_intercept={fit_intercept}, positive={positive}, copy_X={copy_x})\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.neighbors import KNeighborsRegressor\nmodel = KNeighborsRegressor(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.svm import SVR\nmodel = SVR(kernel='{kernel}', degree={degree}, gamma='{gamma}')\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.tree import DecisionTreeRegressor\nmodel = DecisionTreeRegressor(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split})\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split})\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom xgboost import XGBRegressor\nmodel = XGBRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}')\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom lightgbm import LGBMRegressor\nmodel = LGBMRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')\nmodel.fit(X_train, y_train)\n\n"
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
                        progress_bar()
                        st.session_state['trained_model_bool'] = True
            
                        # Train the model
                        st.session_state.all_the_process += f"# Model Building\nfrom catboost import CatBoostRegressor\nmodel = CatBoostRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')\nmodel.fit(X_train, y_train)\n\n"
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
            if not st.session_state.all_the_process_predictions :
                if st.session_state['y_train'] is not None:
                    st.session_state.all_the_process_predictions = True
                    st.session_state.all_the_process += f"# Predictions\ny_pred_train = model.predict(X_train)\n"
                    y_pred_train = model.predict(X_train)
                    st.session_state.y_pred_train = y_pred_train

                if st.session_state['y_val'] is not None:
                    st.session_state.all_the_process += f"y_pred_val = model.predict(X_val)\n"
                    y_pred_val = model.predict(X_val)
                    st.session_state.y_pred_val = y_pred_val

                if st.session_state['y_test'] is not None:
                    st.session_state.all_the_process += f"y_pred_test = model.predict(X_test)\n"
                    y_pred_test = model.predict(X_test)
                    st.session_state.y_pred_test = y_pred_test

            # Choose Evaluation Metric
            if st.session_state['problem_type'] == "Classification":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"], key='evaluation_metric')

            elif st.session_state['problem_type'] == "Regression":
                evaluation_metric = st.multiselect("Evaluation Metric", ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2 Score"], key='evaluation_metric')

            col1, col2, col3 = st.columns([1, 0.6, 1])
            if col2.button("Evaluate Model"):
                st.session_state.show_eval = True
                for metric in evaluation_metric:

                    # if st.session_state['y_train'] is not None :
                    #         st.session_state.all_the_process += f"train_score = accuracy_score(y_train, y_pred_train)\n"
                    
                    if metric == "Accuracy":
                        st.session_state.all_the_process += f"\n# Evaluation - Accuracy \nfrom sklearn.metrics import accuracy_score\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import accuracy_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None :
                            st.session_state.all_the_process += f"train_score = accuracy_score(y_train, y_pred_train)\n"
                            train_score = accuracy_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = accuracy_score(y_val, y_pred_val)\n"
                            val_score = accuracy_score(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None:
                            st.session_state.all_the_process += f"test_score = accuracy_score(y_test, y_pred_test)\n"
                            test_score = accuracy_score(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "Precision":
                        st.session_state.all_the_process += f"\n # Evaluation - Precision\nfrom sklearn.metrics import precision_score\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import precision_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None :
                            st.session_state.all_the_process += f"train_score = precision_score(y_train, y_pred_train)\n"
                            train_score = precision_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = precision_score(y_val, y_pred_val)\n"
                            val_score = precision_score(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None :
                            st.session_state.all_the_process += f"test_score = precision_score(y_test, y_pred_test)\n"
                            test_score = precision_score(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "Recall":
                        st.session_state.all_the_process += f"\n# Evaluation - Recall\nfrom sklearn.metrics import recall_score\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import recall_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None:
                            st.session_state.all_the_process += f"train_score = recall_score(y_train, y_pred_train)\n"
                            train_score = recall_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None:
                            st.session_state.all_the_process += f"val_score = recall_score(y_val, y_pred_val)\n"
                            val_score = recall_score(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None :
                            st.session_state.all_the_process += f"test_score = recall_score(y_test, y_pred_test)\n"
                            test_score = recall_score(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "F1 Score":
                        st.session_state.all_the_process += f"\n# Evaluation - F1 Score\nfrom sklearn.metrics import f1_score\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import f1_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None:
                            st.session_state.all_the_process += f"train_score = f1_score(y_train, y_pred_train)\n"
                            train_score = f1_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None:
                            st.session_state.all_the_process += f"val_score = f1_score(y_val, y_pred_val)\n"
                            val_score = f1_score(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None:
                            st.session_state.all_the_process += f"test_score = f1_score(y_test, y_pred_test)\n"
                            test_score = f1_score(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "AUC Score":
                        st.session_state.all_the_process += f"\n# Evaluation - AUC Score\nfrom sklearn.metrics import roc_auc_score\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import roc_auc_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None:
                            st.session_state.all_the_process += f"train_score = roc_auc_score(y_train, y_pred_train)\n"
                            train_score = roc_auc_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = roc_auc_score(y_val, y_pred_val)\n"
                            val_score = roc_auc_score(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None:
                            st.session_state.all_the_process += f"test_score = roc_auc_score(y_test, y_pred_test)\n"
                            test_score = roc_auc_score(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df
                        
                    elif metric == "Mean Absolute Error (MAE)":
                        st.session_state.all_the_process += f"\n# Evaluation - MAE\nfrom sklearn.metrics import mean_absolute_error\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import mean_absolute_error
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None :
                            st.session_state.all_the_process += f"train_score = mean_absolute_error(y_train, y_pred_train)\n"
                            train_score = mean_absolute_error(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = mean_absolute_error(y_val, y_pred_val)\n"
                            val_score = mean_absolute_error(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None :
                            st.session_state.all_the_process += f"test_score = mean_absolute_error(y_test, y_pred_test)\n"
                            test_score = mean_absolute_error(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "Mean Squared Error (MSE)":
                        st.session_state.all_the_process += f"\n# Evaluation - MSE\nfrom sklearn.metrics import mean_squared_error\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import mean_squared_error
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None :
                            st.session_state.all_the_process += f"train_score = mean_squared_error(y_train, y_pred_train)\n"
                            train_score = mean_squared_error(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = mean_squared_error(y_val, y_pred_val)\n"
                            val_score = mean_squared_error(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None :
                            st.session_state.all_the_process += f"test_score = mean_squared_error(y_test, y_pred_test)\n"
                            test_score = mean_squared_error(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df

                    elif metric == "Root Mean Squared Error (RMSE)":
                        st.session_state.all_the_process += f"\n# Evaluation - RMSE\nfrom sklearn.metrics import mean_squared_error\nfrom math import sqrt\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import mean_squared_error
                        from math import sqrt
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None :
                            st.session_state.all_the_process += f"train_score = sqrt(mean_squared_error(y_train, y_pred_train))\n"
                            train_score = sqrt(mean_squared_error(y_train, y_pred_train))
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = sqrt(mean_squared_error(y_val, y_pred_val))\n"
                            val_score = sqrt(mean_squared_error(y_val, y_pred_val))
                        if st.session_state['y_test'] is not None :
                            st.session_state.all_the_process += f"test_score = sqrt(mean_squared_error(y_test, y_pred_test))\n"
                            test_score = sqrt(mean_squared_error(y_test, y_pred_test))

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df
                        
                    elif metric == "R2 Score":
                        st.session_state.all_the_process += f"\n# Evaluation - R2 Score\nfrom sklearn.metrics import r2_score\ny_pred_train = None, y_pred_test = None, y_pred_val = None\n"
                        from sklearn.metrics import r2_score
                        train_score, test_score, val_score = None, None, None
                        if st.session_state['y_train'] is not None :
                            st.session_state.all_the_process += f"train_score = r2_score(y_train, y_pred_train)\n"
                            train_score = r2_score(y_train, y_pred_train)
                        if st.session_state['y_val'] is not None :
                            st.session_state.all_the_process += f"val_score = r2_score(y_val, y_pred_val)\n"
                            val_score = r2_score(y_val, y_pred_val)
                        if st.session_state['y_test'] is not None :
                            st.session_state.all_the_process += f"test_score = r2_score(y_test, y_pred_test)\n"
                            test_score = r2_score(y_test, y_pred_test)

                        st.session_state.all_the_process += f"lst = [train_score, val_score, test_score]\nnew_lst = [i for i in lst if i != None]\nmetrics_df['{metric}'] = new_lst\n"
                        lst = [train_score, val_score, test_score]
                        new_lst = [i for i in lst if i != None]
                        metrics_df[metric] = new_lst
                        st.session_state['metrics_df'] = metrics_df


            # Show Evaluation Metric
            if show_eval:
                new_line()
                col1, col2, col3 = st.columns([0.5, 1, 0.5])
                st.markdown("### Evaluation Metric")

                if is_train and is_test and is_val:
                    st.session_state.all_the_process += f"# Evaluation Metric\nmetrics_df.index = ['Train', 'Validation', 'Test']\nprint(metrics_df)\n\n"
                    st.session_state['metrics_df'].index = ['Train', 'Validation', 'Test']
                    st.write(st.session_state['metrics_df'])

                elif is_train and is_test:
                    st.session_state.all_the_process += f"# Evaluation Metric\nmetrics_df.index = ['Train', 'Test']\nprint(metrics_df)\n\n"
                    st.session_state['metrics_df'].index = ['Train', 'Test']
                    st.write(st.session_state['metrics_df'])

                elif is_train and is_val:
                    st.session_state.all_the_process += f"# Evaluation Metric\nmetrics_df.index = ['Train', 'Validation']\nprint(metrics_df)\n\n"
                    st.session_state['metrics_df'].index = ['Train', 'Validation']
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
                    
                    if is_test:
                        # Show the ROC curve plot without any columns
                        col1, col2, col3 = st.columns([0.2, 1, 0.2])
                        fig, ax = plt.subplots()
                        plot_roc_curve(model, X_test, y_test, ax=ax)
                        col2.pyplot(fig)

                    elif is_val:
                        # Show the ROC curve plot without any columns
                        col1, col2, col3 = st.columns([0.2, 1, 0.2])
                        fig, ax = plt.subplots()
                        plot_roc_curve(model, X_val, y_val, ax=ax)
                        col2.pyplot(fig)
                        
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
                
                
    col1, col2, col3, col4= st.columns(4)        

    if col1.button("Show df", use_container_width=True):
        st.dataframe(df)

    st.session_state.df.to_csv("df.csv", index=False)
    df_file = open("df.csv", "rb")
    df_bytes = df_file.read()
    if col2.download_button("Download df", df_bytes, "df.csv", key='save_df', use_container_width=True):
        pass

    if col3.button("Show The Code", use_container_width=True):
        st.code(st.session_state.all_the_process, language='python')

    if col4.button("Reset", use_container_width=True):
        st.session_state.clear()
        st.experimental_rerun()


    


