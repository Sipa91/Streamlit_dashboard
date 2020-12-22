# This is the main page
import Home
import Project_Overview
import Results
import Ad_analysis 
import About_us 
import References
import streamlit as st

PAGES = {
    "Home": Home,
    "Project Overview": Project_Overview,
    "Results": Results,
    "Ad Analysis": Ad_analysis,
    "About us": About_us,
    "References": References
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.page()