
import streamlit as st
import requests
import PIL.Image as Image


def new_line():
    st.markdown("<br>", unsafe_allow_html=True)


# Define a function to load the Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Config
page_icon = Image.open("./assets/icon.png")
st.set_page_config(layout="centered", page_title="Click ML", page_icon=page_icon)


# Create the About page
def main():
    # Title Page
    st.markdown("<h1 align='center'> 🔎About", unsafe_allow_html=True)
    new_line()

    # What is ClickML?
    st.markdown("Welcome to ClickML, an intuitive and powerful machine learning application designed to simplify the process of building and evaluating machine learning models. Whether you're a beginner or an experienced data scientist, ClickML provides a user-friendly interface to streamline your machine learning workflows.", unsafe_allow_html=True)
    st.markdown("It is no-code easy-to-use platfrom which allows you to build machine learning models without writing a single line of code. \n ")
    
    # Show Video Prom
    video_path = "./assets/Promo.mp4"
    video_bytes = open(video_path, "rb").read()
    st.video(video_bytes)
    
    
    # what this app does with the main, quickml, and study_time pages
    st.markdown("This app is divided into three main tabs: **👉 ClickML**, **🚀 QuickML**, and **📚 StudyML**.", unsafe_allow_html=True)
    st.write("\n")

    # ClickML
    st.markdown("### 👉 ClickML")
    st.markdown("- **ClickML:** This section is the main page of the **ClickML** web app. It provides the customizability to build Machine Learning models by selecting and applying the Data Preparation techniques that fits your data. Also, you can try differnet Machine Learning models and tune the hyperparameters to get the best model.", unsafe_allow_html=True)
    st.write("\n")

    # QuickML
    st.markdown("### 🚀 QuickML")
    st.markdown("- **QuickML:** QuickML is a tab that allows you to build a model quickly with just a few clicks. This tab is designed for people who are new to Machine Learning and want to build a model quickly without having to go through the entire process of Exploratory Data Analysis, Data Cleaning, Feature Engineering, etc. It is just a quick way to build a model for testing purposes.", unsafe_allow_html=True)
    st.write("\n")

    # StudyML
    st.markdown("### 📚 StudyML")
    st.markdown("- **Study Time:** The StudyML tab is designed to help you to understand the key concepts of building machine learning models. This tab has 7 sections, each section talk about a specific concept in building machine learning models. With each section you will have the uplility to apply the concepts of this sections on multiple datasets. The code the Explaination and everything you need to understand is in this tab.", unsafe_allow_html=True)
    new_line()

    # Why ClickML?
    st.header("✨ Why Choose ClickML?")
    st.markdown("""
- **User-Friendly Interface**: ClickML offers an intuitive and easy-to-use interface, making machine learning accessible to users of all skill levels.
- **Efficiency and Speed**: With ClickML, you can quickly build, train, and evaluate machine learning models, reducing the time and effort required.
- **Comprehensive Learning Resources**: The StudyML tab provides detailed explanations, code examples, and visualizations to enhance your understanding of machine learning concepts.
- **Flexible and Customizable**: ClickML supports a wide range of algorithms and allows you to fine-tune model parameters to meet your specific requirements.

                """, unsafe_allow_html=True)
    new_line()


    # How to use ClickML?
    st.header("📝 How to Use ClickML?")
    st.markdown("Below is a video that explains how to use ClickML by building a machine learning model on the Titanic dataset and by using all the features of ClickML.", unsafe_allow_html=True)
    st.video("./assets/Tutorial.mp4")

    # Contributors
    st.header(" 👤 Contributors")
    st.markdown("This application was developed and maintained by **Basel Mathar**.", unsafe_allow_html=True)
    st.markdown("Basel is a Data Scientist and a Machine Learning Engineer. He is passionate about building Machine Learning models and creating web apps to help others build Machine Learning models.", unsafe_allow_html=True)
    new_line()



    # Source Code
    st.header(" 📂 Source Code")
    st.markdown("The source code for this app is available on [**GitHub**](https://github.com/baselhusam/clickml). Feel free to contribute, provide feedback, or customize the application to suit your needs.", unsafe_allow_html=True)
    st.markdown("You can open the terminal and run the following commands to download the source code and run the app locally:", unsafe_allow_html=True)
    st.code("""git clone https://github.com/baselhusam/ClickML.git
pip install -r requirements.txt
streamlit run 1_ClickML.py""", 
language="bash")
    new_line()

    # Roadmap 
    st.header(" 🗺️ Roadmap")
    st.markdown("""This is a roadmap for the ClickML project. It will show the current status of the project and the future work that needs to be done.
    Visit the [**ClickML Roadmap**](https://clickml-roadmap.streamlit.app/) for more information.""", unsafe_allow_html = True)
    new_line()

    # Contact Us
    st.header(" 💬 Contact Us")
    st.markdown("""If you have any questions or suggestions, please feel free to contact us at **baselmathar@gmail.com**. We're here to help!
    
**Connect with us on social media:** 

<a href="https://www.linkedin.com/company/clickml/?viewAsMember=true" target="_blank">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe0adDoUGWVD3jGzfT8grK5Uhw0dLXSk3OWJwZaXI-t95suRZQ-wPF7-Az6KurXDVktV4&usqp=CAU" alt="LinkedIn" width="80" height="80" style="border-radius: 25%;">
</a>  󠁪 󠁪 󠁪 󠁪 󠁪 
<a href="https://www.instagram.com/baselhusam/" target="_blank">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" alt="Instagram" width="80" height="80" style="border-radius: 25%;">
</a>  󠁪 󠁪 󠁪 󠁪 󠁪 
<a href="https://www.facebook.com/profile.php?id=100088667931989" target="_blank">
  <img src="https://seeklogo.com/images/F/facebook-logo-C64946D6D2-seeklogo.com.png" alt="Facebook" width="80" height="80" style="border-radius: 25%;">
</a>

<br> 
<br>

We look forward to hearing from you and supporting you on your machine learning journey!

    
    """, unsafe_allow_html=True)
                

if __name__ == "__main__":
    main()
