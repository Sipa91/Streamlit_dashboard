#Underpage1
import streamlit as st
from PIL import Image


def page():
    st.title('Apply or not to apply')
    st.subheader('A machine learning approach to predict the appealingness of job ads for female applicants*')
    image = Image.open('Images/Intro.jpg')
    st.image(image, use_column_width=True)
    st.write("""
    Within times of shortage of skilled workers, businesses of all sizes have to think about strategies how to get access to skilled workers and how to attract them.
    Furthermore, times have changed and society as well as politics are interested in increasing the proportion of women working in all kind of industries.

    Thus, companies should focus on attracting women to work for them not only in order to counteract the skilled-worker-shortage, but also for keeping up with the times and being an attractive, modern employer.
    
    Literature suggests, that amongst other things, job advertisements influence the probability of women applying or not. 
    
    Inspired by the existing literature and [TU Munich´s Gender Decoder](https://genderdecoder.wi.tum.de)[1], the idea of this project arised:
    We created a model for predicting whether a german job ad is appealing to women or not - with different job ad features as indicator of appealingness and 
    appealingness again as indicator of womens´ intent to apply.
    
    
    """)
    st.subheader('Guide')
    st.markdown("""
    Analyse your job ad in the section _Ad Analysis_ and find out, if your ad is rather female or male oriented!  

    In the section _Project Overview_, we guide you through the steps we took during our project.    
    
    Get some insights of our job ad exploration in the _Results_ section.    
    
    Last but not least we would like to introduce ourselves - we are happy about feedback and comments :blush:  

    """)  
  
      
    st.markdown("""
    *_Due to the data situation, in our project we only refer to two genders. 
    Information on applicants who do not identify themselves in a binary way is not (yet) included in the data. 
    When we refer to women in this project, we are referring to individuals who identify themselves as female. 
    We look forward to further research that looks at effects of (job) descriptions in a more differentiated way._

    """)

    

   



