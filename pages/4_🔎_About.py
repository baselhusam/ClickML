
import streamlit as st

def new_line():
    st.markdown("<br>", unsafe_allow_html=True)


# Config
st.set_page_config(layout="centered", page_title="Click ML", page_icon="👆")


# Create the About page
def main():
    # Title Page
    st.markdown("<h1 align='center'> 🔎About", unsafe_allow_html=True)
    new_line()
    new_line()

    # What is ClickML?
    st.markdown("ClickML is a web app that is designed to help you build Machine Learning models with just a few clicks. It provides the customizability to build Machine Learning models by selecting and applying the Data Preparation techniques that fits your data. Also, you can try differnet Machine Learning models and tune the hyperparameters to get the best model.", unsafe_allow_html=True)
    
    
    # what this app does with the main, quickml, and study_time pages
    st.markdown("This app is divided into three main sections: **Main**, **QuickML**, and **Study Time**.", unsafe_allow_html=True)
    st.markdown("- **ClickML:** This section is the main page of the **ClickML** web app. It provides the customizability to build Machine Learning models by selecting and applying the Data Preparation techniques that fits your data. Also, you can try differnet Machine Learning models and tune the hyperparameters to get the best model.", unsafe_allow_html=True)
    st.markdown("- **QuickML:** QuickML is a tab that allows you to build a model quickly with just a few clicks. This tab is designed for people who are new to Machine Learning and want to build a model quickly without having to go through the entire process of Exploratory Data Analysis, Data Cleaning, Feature Engineering, etc. It is just a quick way to build a model for testing purposes.", unsafe_allow_html=True)
    st.markdown("- **Study Time:** The StudyML tab is designed to help you to understand the key concepts of building machine learning models. This tab has 7 sections, each section talk about a specific concept in building machine learning models. With each section you will have the uplility to apply the concepts of this sections on multiple datasets. The code the Explaination and everything you need to understand is in this tab.", unsafe_allow_html=True)
    new_line()

    # Why ClickML?
    st.markdown("### 🖱️ Why ClickML?")
    st.markdown("ClickML is designed to help you build Machine Learning models with just a few clicks. It provides the customizability to build Machine Learning models by selecting and applying the Data Preparation techniques that fits your data. Also, you can try differnet Machine Learning models and tune the hyperparameters to get the best model.", unsafe_allow_html=True)
    new_line()

    # Contributors
    st.markdown("### 👤 Contributors")
    st.markdown("This app was created by **Basel Mathar**.", unsafe_allow_html=True)
    st.markdown("Basel is a Data Scientist and a Machine Learning Engineer. He is passionate about building Machine Learning models and creating web apps to help others build Machine Learning models.", unsafe_allow_html=True)
    new_line()



    # Source Code
    st.markdown("### 📂 Source Code")
    st.markdown("The source code for this app is available on [**GitHub**](https://github.com/baselhusam/clickml).", unsafe_allow_html=True)
    st.markdown("Open the terminal and run the following commands to download the source code and run the app locally:", unsafe_allow_html=True)
    st.code("""git clone https://github.com/baselhusam/ClickML.git
pip install -r requirements.txt
streamlit run 1_👉_ClickML.py""", 
language="bash")
    new_line()

    # Contact Us
    st.markdown("### 💬 Contact Us")
    st.markdown("If you have any questions or suggestions, please feel free to contact us at **baselmathar@gmail.com**.", unsafe_allow_html=True)
                


    
    

if __name__ == "__main__":
    main()
