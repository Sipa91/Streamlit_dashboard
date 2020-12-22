# Underpage3
import streamlit as st
from PIL import Image


def page():
    st.title('About us')

    col1, col2 = st.beta_columns(2)
    with col1:
        image = Image.open('Images/Petra.jpg')
        st.subheader('Dr. Petra Pinger')
        st.image(image, width=180)
        st.write("""
        As a psychologist and educational researcher I was always interested in answering questions with data.
        Building on my experience in research and project management combined with the tech skills and knowledge about machine learning I acquired in the bootcamp,
        I am ready for a new career in the exciting field of data science.  
        You can find my whole profile [here](https://talents.neuefische.com/student/16d4f3d6-5b41-4c13-81ca-561f8fd518dc).
        """)
        st.subheader('Contact')
        st.write("""
        [LinkedIn](https://www.linkedin.com/in/dr-petra-pinger-19540b1a6/)  
        [Xing](https://www.xing.com/profile/Petra_Pinger4/cv)
        """)

    with col2:
        with col2:
            image = Image.open('Images/Sina.jpg')
            st.subheader('Sina Pietrowski')
            st.image(image, width=260)
            st.write("""As a psychologist I have extensive experience in data collection 
            and data analysis. Scientific and experimental work as well as comprehensive statistics 
            were the main focus of my studies as well as in various stages of my professional career. 
            During the Data Science Bootcamp, I could deepen and expand my knowledge and test it directly in practical projects. 
            Now, I am ready to start as a Data Scientist or Data Analyst!  
            You can find my whole profile [here](https://talents.neuefische.com/student/593e4a7e-a513-41f0-8ce4-44353144e38a).
            """)
            st.subheader('Contact')
            st.write("""
            [LinkedIn](https://www.linkedin.com/in/sina-pietrowski-2b570519a)  
            [Xing](https://www.xing.com/profile/Sina_Pietrowski/cv)
            """)

        
    
    



