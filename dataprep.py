import pandas as pd
from sklearn import preprocessing
from scipy import stats
import ast
import re, string, unicodedata
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from wordlists import workingtime_words, feminine_coded_words, masculine_coded_words, salary_words, workingtime_words, profit_words, family_words, homeoffice_words, safety_words, health_words, travel_words
#from dataprep import *


# Functions for Text-Data preparation and extracting gender-coded words

def strip_html(text):
    text = text.replace('&nbsp;', ' ') # replace html space specifier with space char
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(' ', strip=False)  # space between tags, don't strip newlines

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_punctuation(string, char_list, replacer, regex = r"[.,;:–!•]"):

    for char in char_list:
        string = string.replace(char, replacer)
    string_cleaned = re.sub(regex, "", string, 0)
    return string_cleaned

def clean_ad(text):
    ''' removes html-stuff and replaces parantheses with spaces '''
    text_denoised = denoise_text(text)
    text_depunct = remove_punctuation(text_denoised, ['(', ')', '"'], " ")
    return text_depunct

def tokanize(text):
    ''' seperates a text and puts words into a list, 
    lowercases words and removes "empty"-list items. 
    Returns a list of all words '''
    # Lowercasing text
    text_lower = text.lower()
    # Split text into tokens
    tokens = re.split('\s', text_lower)
    #remove "empty tokens"
    tokens_full = list(filter(None, tokens))
    return tokens_full


def add_processing_columns(df):
    df['ad_cleaned'] = df.apply(lambda x: clean_ad(x['description']), axis=1)
    df['ad_tokens'] = df.apply(lambda x: tokanize(x['ad_cleaned']), axis=1)
    return df

    # add column "ad_length" which is the number of words in ad
def ad_length (df):
    df['ad_length'] = df.apply(lambda x: len(x['ad_tokens']), axis=1)
    return df

#_______________________________________________________________________________________________________________________

# Gender-coded words

# Function that checks if ads contains gendered words and returns the gendered-coded words 
# and the number of gender-coded words. This funtion has been copied/adapted from German "Decoder" developed by TU München 
# please find reference in README 
def find_and_count_coded_words(advert_word_list, gendered_word_list):
    gender_coded_words = [word for word in advert_word_list
        for coded_word in gendered_word_list
        if coded_word in word]
        # if word.startswith(coded_word)]
    return list(gender_coded_words), len(gender_coded_words) 

def add_genderword_columns(df):
    # add column "words_f". contains list a feminine-associated words
    df['words_f'] = df.apply(lambda x: find_and_count_coded_words(x['ad_tokens'], feminine_coded_words)[0], axis=1)
    # add column "words_m". contains list a masculine-associated words
    df['words_m'] = df.apply(lambda x: find_and_count_coded_words(x['ad_tokens'], masculine_coded_words)[0], axis=1)
    # add column "words_f_count". contains number of feminine-associated words
    df['words_f_count'] = df.apply(lambda x: find_and_count_coded_words(x['ad_tokens'], feminine_coded_words)[1], axis=1)
    # add column "words_m". contains number of masculine-associated words
    df['words_m_count'] = df.apply(lambda x: find_and_count_coded_words(x['ad_tokens'], masculine_coded_words)[1], axis=1)
    return df

#_______________________________________________________________________________________________________________________

# Functions for preparing dataset:

def transform_ID (df):

    # Creating dictionary, matching discipline_id with discipline label
    disciplines = {1008:'Health_Medical_Social',
    1015:'PR_Journalism ',
    1020:'Law',
    1022:'Other_Disciplines',
    1001:'Analysis_Statistics',
    1002:'Administration ',
    1003:'Consulting',
    1004:'Customer Service',
    1005:'Purchasing_Materials_Management_Logistics',
    1006:'Finance_Accounting_Controlling',
    1007:'Teaching_R&D',
    1009:'Graphic_Design_Architecture',
    1010:'Engineering_Technical',
    1011:'IT_Software_Development',
    1012:'Management_Corporate_Development',
    1013:'Marketing_Advertising',
    1014:'HR',
    1016:'Production_Manufacturing',
    1017:'Product_Management',    
    1018:'Project_Management',
    1019:'Process_Planning_QA',
    1021:'Sales_Commerce'               
    }

    # Creating dictionary, matching industry_id with industry label
    industries = {0:'Other',
    10000:'Architecture_planning',
    20000:'Consumer_goods_trade',
    30000:'Automotive_vehicle_manufacturing',
    40000:'Industry_mechanical_engineering ',
    50000:'Medical_services',
    60000:'Energy_water_environment',
    70000:'Transport_logistics',
    80000:'Tourism_food_service',
    90000:'Internet_IT',
    100000:'Telecommunication',
    110000:'Media_publishing',
    120000:'Banking_financial_services',
    130000:'Insurance',
    140000:'Real_Estate',
    150000:'Auditing_tax_law ',
    160000:'Consulting',
    170000:'Marketing_PR_design ',
    180000:'HR_services ',    
    190000:'Civil_service_associations_institutions',
    200000:'Education_science ',
    210000:'Health_social',  
    220000:'Art_culture_sport',
    230000:'Other'
    }

    df['discipline_label'] = df['discipline_id'].map(disciplines)
    df['industry_label'] = df['industry_id'].map(industries)
     
    df['discipline_label'].astype("category")
    df['industry_label'].astype("category")

    return df


# Categorizing industries into male and female dominated
def gender_dominance_ind(df):
    female_ratio_for_industry = df["female_ratio_for_industry"]

    gender_dominance_ind = []
    for i in female_ratio_for_industry:
        if i >= 0.5:
            gender_dominance_ind.append("female")
        else:
            gender_dominance_ind.append("male")
            
    # Creating new column
    df["gender_dominance_ind"] = gender_dominance_ind

    # Changing data type of newly created column "gender_dominance_ind" into category
    df["gender_dominance_ind"].astype("category")

    return df

# Categorizing disciplines into male and female dominated
def gender_dominance_dis(df):
    
    female_ratio_for_discipline = df["female_ratio_for_discipline"]

    gender_dominance_dis = []
    for i in female_ratio_for_discipline:
        if i >= 0.5:
            gender_dominance_dis.append("female")
        else:
            gender_dominance_dis.append("male")

    # Creating new column
    df["gender_dominance_dis"] = gender_dominance_dis

    # Changing data type of newly created column "gender_dominance_dis" into category
    df["gender_dominance_ind"].astype("category")

    return df


def drop_columns(df, list_columns):
    '''Drops columns in the list and returns updated dataframe'''
    df_new = df.drop(list_columns, axis=1)
    return df_new

def balancing(df):
    count_not_appealing, count_appealing = df["jobad_appealingness_dis"].value_counts()
    # Shuffle the Dataset.
    shuffled_df = df.sample(frac=1,random_state=42)
    # Put all the appealing ads in a separate dataset.
    appealing_df = shuffled_df.loc[shuffled_df["jobad_appealingness_dis"] == 1]
    #Randomly select 4914 observations from the not appealing (majority class)
    not_appealing_df = shuffled_df.loc[shuffled_df["jobad_appealingness_dis"] == 0].sample(n=count_appealing, random_state=42)

    # Concatenate both dataframes again
    balanced_df = pd.concat([appealing_df, not_appealing_df])
    balanced_df =balanced_df.reset_index(drop = True)
    return balanced_df

# Preparation for building a word count dictionary for lemmatized words
def unpack_string (column):
    column = ast.literal_eval(column)
    column = [i.strip() for i in column]
    column = " ".join(column)
    return column

def add_lemmatized_unpacked_column(df):
    df['lemmatized_unpacked'] = df.apply(lambda x: unpack_string(x['ad_lemmatized']), axis=1)
    return df

#_______________________________________________________________________________________________________________________

# Functions for extracting salary and other benefits

def find_salary (advert_word_list, salary_word_list, profit_word_list):
    second_half = advert_word_list[int((len(advert_word_list))/2):]
    salary_words = [coded_word for coded_word in salary_word_list
        for word in profit_word_list if coded_word in second_half
        and word not in second_half]
    euro = ["€" for word in second_half for profit in profit_word_list if "€" in word and profit not in second_half]
    number = [number for number in second_half if re.match('\d{3}', number) is not None]
    salary = 1 if len(salary_words) != 0 and number or len(euro) != 0 and number else 0 
    return salary_words, euro, number, salary

def add_salary_columns(df):
    df['salary_words'] = df.apply(lambda x: find_salary(x['ad_tokens'], salary_words, profit_words)[0], axis=1)
    df['euro'] = df.apply(lambda x: find_salary(x['ad_tokens'], salary_words, profit_words)[1], axis=1)
    df['number'] = df.apply(lambda x: find_salary(x['ad_tokens'], salary_words, profit_words)[2], axis=1)
    df['salary'] = df.apply(lambda x: find_salary(x['ad_tokens'], salary_words, profit_words)[3], axis=1)
    return df

def find_workingtime(advert_word_list, workingtime_word_list):
    workingtime_words = [word for word in advert_word_list
        for coded_word in workingtime_word_list
        if coded_word in word]
    workingtime = len(workingtime_words)
    return workingtime_words, workingtime

def add_workingtime_columns(df):
    df['workingtime_words'] = df.apply(lambda x: find_workingtime(x['ad_tokens'], workingtime_words)[0], axis=1)
    df['workingtime_info'] = df.apply(lambda x: find_workingtime(x['ad_tokens'], workingtime_words)[1], axis=1)
    return df 

def find_family_benefits(advert_word_list, family_word_list):
    family_words = [word for word in advert_word_list
        for coded_word in family_word_list
        if coded_word in word]
    family_benefits = len(family_words)
    return family_words, family_benefits

def add_family_columns(df):
    df['family_words'] = df.apply(lambda x: find_family_benefits(x['ad_tokens'], family_words)[0], axis=1)
    df['family_benefits'] = df.apply(lambda x: find_family_benefits(x['ad_tokens'], family_words)[1], axis=1)
    return df 

def find_homeoffice(advert_word_list, homeoffice_word_list):
    homeoffice_words = [word for word in advert_word_list
        for coded_word in homeoffice_word_list
        if coded_word in word]
    homeoffice = len(homeoffice_words)
    return homeoffice_words, homeoffice

def add_homeoffice_columns(df):
    df['homeoffice_words'] = df.apply(lambda x: find_homeoffice(x['ad_tokens'], homeoffice_words)[0], axis=1)
    df['homeoffice_opportunity'] = df.apply(lambda x: find_homeoffice(x['ad_tokens'], homeoffice_words)[1], axis=1)
    return df

def find_safety_words(advert_word_list, safety_word_list):
    safety_words = [word for word in advert_word_list
        for coded_word in safety_word_list
        if coded_word in word]
    safety = len(safety_words)
    return safety_words, safety

def add_safety_columns(df):
    df['safety_words'] = df.apply(lambda x: find_safety_words(x['ad_tokens'], safety_words)[0], axis=1)
    df['safety_statements'] = df.apply(lambda x: find_safety_words(x['ad_tokens'], safety_words)[1], axis=1)
    return df

def find_health_words(advert_word_list, health_word_list):
    health_words = [word for word in advert_word_list
        for coded_word in health_word_list
        if coded_word in word]
    health = len(health_words)
    return health_words, health

def add_health_column(df):
    df['health_words'] = df.apply(lambda x: find_health_words(x['ad_tokens'], health_words)[0], axis=1)
    df['health_benefits'] = df.apply(lambda x: find_health_words(x['ad_tokens'], health_words)[1], axis=1)
    return df


# Function that checks if the ad contains traveling requirements
def find_travel_words (advert_word_list, travel_word_list):
    travel_words = [word for word in advert_word_list
        for coded_word in travel_word_list
        if coded_word in word]
    travel = len(travel_words)
    return travel_words, travel

def add_traveling_columns(df):
    df['travel_words'] = df.apply(lambda x: find_travel_words(x['ad_tokens'], travel_words)[0], axis=1)
    df['traveling'] = df.apply(lambda x: find_travel_words(x['ad_tokens'], travel_words)[1], axis=1)
    return df

#_______________________________________________________________________________________________________________________ 
# Define target variable

def add_target(df):
    df = df.eval("relative_female_ratio_discipline = female_ratio_of_applicants/female_ratio_for_discipline")
    df = df.eval("relative_female_ratio_industry = female_ratio_of_applicants/female_ratio_for_industry")
    return df

def categorize_target(df):
    # Creating new column relative female ratio discipline
    df["jobad_appealingness_dis"] = np.where(ads["relative_female_ratio_discipline"] < 1, 0, 1)

    # Changing data type of newly created column "jobad_appealingness_dis" into category
    df["jobad_appealingness_dis"] = df["jobad_appealingness_dis"].astype("category")

    # Creating new column relative female ratio discipline
    df["jobad_appealingness_ind"] = np.where(ads["relative_female_ratio_industry"] < 1, 0, 1)

    # Changing data type of newly created column "jobad_appealingness_dis" into category
    df["jobad_appealingness_dis"] = df["jobad_appealingness_ind"].astype("category")
    return df
#_______________________________________________________________________________________________________________________    
    
# Functions for creating wordclouds

# Step 1: Function, which creates a string out of the single lists in the columns words_f and words_m
def create_string (column):
    ls_merged = []
    for ls in column:
        for word in ls:
            ls_merged.append(word)
        string = ' '.join(ls_merged)
    return string

# Step 2: Function for creating a wordcloud
def make_wordcloud (string):
    plt.figure(figsize=(15,10))
    wordcloud = WordCloud(background_color="white", width=500, height=250, max_words = 20, collocations = False).generate(string);
    plt.imshow(wordcloud, interpolation="bilinear");
    plt.axis("off");
    return wordcloud

#_______________________________________________________________________________________________________________________

# Function to detect the language of a text 

def detect_lang(ad):
    '''this function detects the language of a text and return either English, German or Others'''
    try:
        detect(ad)
        if detect(ad) == "en":
            return 'English'
        elif detect(ad) == "de":
            return 'German'
        else:
            return 'Others'
    except Exception:
        pass


#_______________________________________________________________________________________________________________________

# Function to calculate correlations
def correlation (column1, column2):
    corr = stats.pearsonr(column1, column2)[0]
    p = stats.pearsonr(column1, column2)[1]
    
    result = f"The correlation between {column1.name} and {column2.name} is {corr:.2f} and the p-value is {p:.2f}"
    return result


#_______________________________________________________________________________________________________________________

# Functions for Text-Data preparation with Spacy
# Using NLP pipeline for lemmatization, POS-tagging and entity extraction
#To use the language pipe to stream texts, single functions are defined which directly work on a spaCy Doc objects. 
#These functions are then called in batches to work on a sequence of Doc objects that are streamed through the pipe.

# python -m spacy download de_core_news_sm

def set_up_spacy():
    model_de = spacy.load("de_core_news_sm")
    nltk.download('stopwords')
    german_stop_words = set(stopwords.words('german'))
    return model_de, german_stop_words


# Single functions for lemmatization, POS-Tagging as well as entitiy extraction
def lemmatize_pipe (text):
    '''Creates list with lowercased, lemmatized tokens without stopwords, numbers and empty string'''
    tokens_lem = [word.lemma_.lower() for word in text if word.text.lower() not in german_stop_words and len(word.text.lower())>1 and 
                  word.text.lower().isalpha()]
    
    return tokens_lem

def postag_pipe (text):  
    """Get POS-Tags for all words"""
    pos_tags = [word.tag_ for word in text]
    
    return pos_tags

def entity_pipe (text):  
    """Get entities out of ads"""
    entities = [(word.text, word.label_) for word in text.ents]
    
    return entities

# Single functions are called in batches within the three following functions
def preprocess_pipe_lem(texts):
    preproc_lem = []
    for doc in model_de.pipe(texts, batch_size=5000, disable=["tagger", "parser", "ner"]):
        preproc_lem.append(lemmatize_pipe(doc))
        
    return preproc_lem

def preprocess_pipe_postag(texts):
    preproc_postag = []
    for doc in model_de.pipe(texts, batch_size=5000, disable=["parser", "ner"]):
        preproc_postag.append(postag_pipe(doc))
        
    return preproc_postag

def preprocess_pipe_ent(texts):
    preproc_ent = []
    for doc in model_de.pipe(texts, batch_size=5000, disable=["tagger", "parser"]):
        preproc_ent.append(entity_pipe(doc))
        
    return preproc_ent

def add_spacy_columns(df):
    df['ad_lemmatized'] = preprocess_pipe_lem(df["ad_cleaned"])
    df['ad_postag'] = preprocess_pipe_postag(df["ad_cleaned"])
    df['ad_entities'] = preprocess_pipe_ent(df["ad_cleaned"])
    return df