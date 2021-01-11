#Underpage3
import streamlit as st
from PIL import Image


def page():
    st.title('Results')
    st.header("""
    Global insights into the world of job applications
    """)
    st.markdown("""
    - 6 out of the 22 different disciplines as examples
    - _HR_, _Marketing_ and _Health, Medical & Social_ as stereotypically rather _female_ disciplines
    - _IT & Software Development_ and _Engineering & Technical_ as stereotypically rather _male_ disciplines
    - _Project Management_ as rather neutral discipline
    """)  

    
    st.subheader("Female applicant ratio for different disciplines")
    st.markdown("""    
    The following plot shows the different female applicant ratios for our example disciplines.
    """)  
    st.markdown("**Mean female ratio of applicants per discipline**")
    image = Image.open('Images/fem_ratio_app.png')
    st.image(image, use_column_width=False, output_format='png')
    
    # Appealingness and disciplines
    st.subheader("Job ad appealingness within male- versus female dominated disciplines")
    st.markdown("""
    Explanation: Disciplines are classified as female-dominated if the female ratio is more or equal 50%.
    
    The following plots show:  
    - There is a greater part of not appealing job ads than appealing ones within male dominated disciplines. 
    - There is a greater part of appealing job ads than not appealing ones in female dominated disciplines. 
    - The _IT & Software Development_ discipline seems to be an exception with more appealing than not appealing job ads :female-technologist:  
    """)
    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown("**Male-dominated disciplines**")
        image = Image.open('Images/appealingness_dis_male.png')
        st.image(image, use_column_width=True, output_format='png')

    with col2:
        st.markdown("**Female-dominated disciplines**")
        image = Image.open('Images/appealingness_dis_fem.png')
        st.image(image, use_column_width=True, output_format='png')

    # Male versus female words
    st.subheader("Most frequent female and male words within our job ads")
    col1, col2 = st.beta_columns(2)

    with col1:
        st.subheader("Male words")
        image = Image.open('Images/male_words.png')
        st.image(image, use_column_width=True, output_format='png')
    with col2:
        st.subheader("Female words")
        image = Image.open('Images/female_words.png')
        st.image(image, use_column_width=True, output_format='png')
        
    st.subheader("Number of male and female words in not appealing versus appealing job ads")
    st.markdown(""" 
    The following plots show:  
    - There is a greater part of male words in not appealing job ads than in appealing ones.
    - There is a greater part of female words in appealing job ads than in not appealing ones. 
    """)
    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown("**Male words**")
        image = Image.open('Images/mwords_count.png')
        st.image(image, use_column_width=True, output_format='png')

    with col2:
        st.markdown("**Female words**")
        image = Image.open('Images/fwords_count.png')
        st.image(image, use_column_width=True, output_format='png')


    # Benefits
    st.subheader("Mentioned benefits and appealingness")
    st.markdown(""" 
    The following plots show, that there is a slightly higher number of female-oriented benefits mentioned 
    in appealing job ads than in not appealing ones.
    """)
    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown("**Number of mentioned benefits in not appealing versus appealing job ads**")
        image = Image.open('Images/benefits_count.png')
        st.image(image, use_column_width=True, output_format='png')
    with col2:
        st.markdown("**Most often mentioned female-oriented benefits**")
        image = Image.open('Images/benefits.png')
        st.image(image, use_column_width=True, output_format='png')


