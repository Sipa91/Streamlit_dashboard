#Underpage2
import streamlit as st
from PIL import Image


def page():
    st.title('Project Overview')

    # Describing existing literature and features
    with st.beta_expander("What our idea is based on and what we expected to find"):
        st.subheader('Characteristics which play a role when it comes to job ad appealingness')
        st.markdown("""
        - **Number of female versus male words**  
        Studies found that women feel less addressed by job ads, which contain lots of male words[2,3].     
        The wordlists containing stereotypical female (f.e. _Unterstützung_, _Team_) and male words (f.e. _Führen_, _Stärke_) are kindly provided by TU Munich´s _Lehrstuhl für Forschungs- und Wissenschaftsmanagement_. 
        - **Number of requirements listed in the job ad**  
        According to published surveys, women tend to apply only if they fulfill around 100% of the requirements. Men, however, also apply when they fulfill only around 60% of the stated requirements[4].
        - **Transparent salary**  
        According to published surveys, knowing about the salary is more important to women[4].
        - **Benefit information**  
        For women, information about benefits, regarding f.e. flexible working policies, parental leave or healthcare is more important[4].  
        The wordlists for female-oriented benefits were compiled by us on base of literature research.
        """)

        st.subheader("Our features for predicting job ad appealingness")
    
        image = Image.open('Images/Features.png')
        st.image(image, use_column_width=True)

    # Describing Dataset
    with st.beta_expander("What data we used"):
        
        st.markdown("""
        **11047 job ad descriptions!** :muscle: """)
        st.markdown("**Example ad**")
        image = Image.open('Images/Data.png')
        st.image(image, use_column_width=True)


    #Defining target variable
    with st.beta_expander("How we defined our target variable"):
        st.markdown("""
         **Goal** :dart:  
        Predicting appealingness of a job ad for a female applicant based on features of the job ad.  
        **And how to define _appealingness_?** :thinking_face: 
        We defined a job ad to be appealing if more women than expected (based on female ratio in discipline) applied.   
         """)
        image = Image.open('Images/Target.png')
        st.image(image, use_column_width=True)

        st.subheader('Example')  
        st.markdown("""
        **Appealing job ad**: 60% women applied for a job in a discipline, where the female ratio is 30%.  
        **Not appealing job ad**: Only 20% women applied for a job ad of a discipline, where the female ratio is 60%.  
        
        """)

    # Explaining Modeling
    with st.beta_expander("How we predicted womens´intent to apply"):
        st.subheader("""
        Models we used for classifying appealing and not-appeling job ads
        """)
        image = Image.open('Images/Modeling.png')
        st.image(image, use_column_width=True)
