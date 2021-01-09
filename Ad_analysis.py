# Underpage4
import streamlit as st
#import matplotlib.pyplot as plt
#import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import pandas as pd
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords
import spacy
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from wordlists import workingtime_words, feminine_coded_words, masculine_coded_words, salary_words, workingtime_words, family_words, homeoffice_words, safety_words, health_words, travel_words
from dataprep import strip_html, remove_between_square_brackets, denoise_text, remove_punctuation, clean_ad, tokanize, create_string, make_wordcloud, find_workingtime, find_family_benefits, find_homeoffice, find_safety_words, find_health_words, find_and_count_coded_words, add_genderword_columns, find_salary, add_salary_columns, add_workingtime_columns, add_family_columns, add_homeoffice_columns, add_safety_columns, add_health_column, add_target, categorize_target, add_processing_columns, ad_length, find_travel_words, add_traveling_columns, drop_columns, add_spacy_columns, add_lemmatized_unpacked_column

# Functions which are needed when creating the sub-page
def find_genderwords(jobad):
    tokens =  clean_ad(jobad) # oder #lemmatize_pipe(jobad)
    tokens = tokanize(tokens) 
    female_words = find_and_count_coded_words(tokens, feminine_coded_words)[0]
    male_words = find_and_count_coded_words(tokens, masculine_coded_words)[0]
    set_fem = set(female_words)
    set_mal = set(male_words)
    number_female_words = len(set_fem)
    number_male_words = len(set_mal)
    return female_words, " ".join(set_fem), number_female_words, male_words, " ".join(set_mal), number_male_words

def lemmatize_pipe (text):
    nltk.download('stopwords')
    german_stop_words = stopwords.words('german')
    model_de = spacy.load("de_core_news_sm")
    text = clean_ad(text)
    text = model_de(text)
    '''Creates list with lowercased, lemmatized tokens without stopwords, numbers and empty string'''
    tokens_lem = [word.lemma_.lower() for word in text if word.text.lower() not in german_stop_words and len(word.text.lower())>1 and 
                 word.text.lower().isalpha()]
    return tokens_lem

def find_benefits(jobad):
    tokens = lemmatize_pipe(jobad)
    workingtime = find_workingtime(tokens, workingtime_words)[0]
    family = find_family_benefits(tokens, family_words)[0]
    homeoffice = find_homeoffice(tokens, homeoffice_words)[0]
    safety = find_safety_words(tokens, safety_words)[0]
    health = find_health_words(tokens, health_words)[0]
    benefits = workingtime + family + homeoffice + safety + health
    benefits = set(benefits)
    return " ".join(benefits)

def create_xtest_features(df):
    df = add_processing_columns(df) #ad_tokens, ad_cleaned
    df = ad_length(df) #ad_length
    df = add_genderword_columns(df) #words_f, words_m, words_f_count, words_m_count
    df = add_salary_columns(df) # salary_words, euro, number, salary
    df = add_workingtime_columns(df) #workingtime_words, workingtime_info
    df = add_family_columns(df) # family_words, family_benefits
    df = add_homeoffice_columns(df) # homeoffice_words, homeoffice_opportunity
    df = add_safety_columns(df) #safety_words, safety
    df = add_health_column(df) #health_words, health
    df = add_traveling_columns(df) #travel_words, traveling
    unnecessary_columns = ['ad_cleaned', 'description', 'ad_tokens', 'workingtime_words', 'homeoffice_words',
       'salary_words', 'family_words', 'safety_words', 'health_words','number', 'euro', 'words_f', 'words_m', 'travel_words']
    xtest_features = drop_columns(df, unnecessary_columns)
    return xtest_features

def create_xtest_NLP(df):
    df_NLP = add_processing_columns(df) #ad_tokens, ad_cleaned
    vect_ltf = joblib.load('/models/vect_ltf.pkl')
    xtest_NLP = vect_ltf.transform(df_NLP['ad_cleaned'])
    return xtest_NLP

def create_xtest_final(df, pred1, pred2):
    df_copy = df.copy()
    df_copy["XGB"] = pred1[:, 1]
    df_copy["LogReg_NLP"] = pred2[:, 1]
    X_final = df_copy[["XGB", "LogReg_NLP"]]
    return X_final

def ad_classification (val, dictionary):
    for key, value in dictionary.items():
        if val == value:
            return key

# Creation of Page
def page():
    st.title('Job Ad Analysis')
    # Build input text area
    message = st.text_area("On this page you can try out our analyzer algorithm. Copy and paste a German job ad you are interested in into the space below and see what happens.", height = 180, 
    value = "Junior Recruiter (m/w/d)  Wir sind ein erfolgsorientiertes Unternehmen und suchen für unser Team zum nächstmöglichen Zeitpunkt engagierte und leistungsfähige Unterstützung. Das solltest Du mitbringen: Erfolgreich abgeschlossene Berufsausbildung oder Studium, idealerweise erste praktische Erfahrung im Recruiting oder Personalwesen, Teamfähigkeit sowie strukturierte Arbeitsweise und Kommunikationsfähigkeit.  Benefits: unbefristeter Arbeitsvertrag, Kita Plätze")

    # Build select options on what to analyze
    activities = ["Find gender-coded words", "Find female-oriented benefits", "Predict womens´ intent to apply"]
    choice = st.selectbox("Choose what you want to find out", activities)

    if choice == "Find gender-coded words":
        # Find gender-coded words and print them out
        st.info("Find gender-coded words within your job ad")
        if st.button("Analyze my job ad"):
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown("**Female words within your job ad**")
                gender_words_fem = find_genderwords(message)[1]
                st.success(gender_words_fem)
                st.markdown("**Number**")
                num_gender_words_fem = find_genderwords(message)[2]
                st.success(num_gender_words_fem)
            with col2:    
                st.markdown("**Male words within your job ad**")
                gender_words_male = find_genderwords(message)[4]
                st.success( gender_words_male)
                st.markdown("**Number**")
                num_gender_words_male = find_genderwords(message)[5]
                st.success(num_gender_words_male)
            # Create Wordcloud over all words in job ad
            st.subheader("Wordcloud")
            nltk.download('stopwords')
            german_stop_words = stopwords.words('german')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            text = clean_ad(message)
            wordcloud = WordCloud(stopwords = german_stop_words, background_color="white", width=500, height=250, max_words = 50, min_word_length = 3, relative_scaling = 1, collocations = False).generate(text)
            # Display the generated image:
            with _lock:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()

          

    if choice == "Find female-oriented benefits":
    # Find female-oriented benefits and print them out
        st.info("Find female-oriented benefits mentioned in your job ad")
        if st.button("Analyze my job ad"):
            st.write("Female-oriented benefits within your job ad:")
            benefits = find_benefits(message)
            st.success(benefits)

        

    if choice == "Predict womens´ intent to apply":
    # Predict of a job ad is appelaing to women or not
        st.info("Find out, if women would rather apply for this job ad or not")
        prediction_labels = {"not appealing": 0, "appealing": 1}
        if st.button("Analyze my job ad"):
            # Create Dateframe out of user input
            data = {'description':  [message]}
            df = pd.DataFrame (data, columns = ['description'])
            # Add all relevant columns for the two different X_test
            xtest_features = create_xtest_features(df)
            xtest_NLP = create_xtest_NLP(df)
            
            # Load trained models
             # Load trained models
            xgboost = joblib.load('/models/xgboost.pkl')
            logreg_NLP = joblib.load('/models/logreg_NLP.pkl')
            final_model = joblib.load('/final_model.pkl')


            # Make predictions with thee different pre-trained models
            pred_xg = xgboost.predict_proba(xtest_features)
            pred_logreg_NLP = logreg_NLP.predict_proba(xtest_NLP)

            # Create new Dataframe out of different predictions
            X_final = create_xtest_final(df, pred_xg, pred_logreg_NLP)

            # Make final prediction with aggegated model
            pred_final = final_model.predict(X_final)
            pred_final_proba = final_model.predict_proba(X_final)
            
            # Get label for predicted class
            final_classification = ad_classification(pred_final, prediction_labels)
            st.success(f"Your job ad is rather {final_classification} to women.")

            st.write("Please keep in mind that this prediction is in more than 3.5 out of 5 cases correct.")
           
           
            
    
            



            