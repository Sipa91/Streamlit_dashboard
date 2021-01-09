# Apply or not to apply -  
# A machine learning approach to predict the appealingness of job ads for female applicants  

Within times of shortage of skilled workers, businesses of all sizes have to think about strategies how to get access to skilled workers and how to attract them.
Furthermore, times have changed and society as well as politics are interested in increasing the proportion of women working in all kind of industries.

Thus, companies should focus on attracting women to work for them not only in order to counteract the skilled-worker-shortage, but also for keeping up with the times and being an attractive, modern employer.
    
Literature suggests, that amongst other things, job advertisements influence the probability of women applying or not. 
    
Inspired by the existing literature and [TU Munich´s Gender Decoder](https://genderdecoder.wi.tum.de), the idea of this project arised:
We created a model for predicting whether a German job ad is appealing to women or not - with different job ad features as indicator of appealingness and 
appealingness again as indicator of womens´ intent to apply.

We have optimized some scripts for our interactive WebApp. We created a separate repository for this, which Streamlit.io accesses. 

# Modeling
- We trained and tested several models (see README file and notebooks in modeling folder for more details)
- Using the features (e.g. benefits, gender words) we extracted from the ad descriptions we trained several classification models. In the end we chose the best model in terms of accuracy (XGBoost).
- Furthermore, we trained different NLP models on the whole lemmatized and (by TfidfVectorizer from Scikit-learn) vectorized job ads. Again we chose the best model in terms of accuracy (logistic regression)
- Finally, we used the predictions of the the models as features for training our final model, a logistic regression.
- Our final model´s predictions has an accuracy of above 70%.
You can find more information about data preprocessing and modeling [here](https://github.com/PetraPi/datascience-Capstone_Job_Ads).

# Webapp & Deployment
We used Streamlit to create this Webapp. Within this application you can copy and paste a self chosen German job ad and see which female versus male words and which female-oriented benefits are included, as well as your job ad is rather appealing to women or not.  


# Acknowledgement
We would like to thank Prof. Dr. Claudia Peus, Regina Dutz and the entire research team at TU Munich for providing the gender word lists and the source code of the decoder.     
      
We would also like to point out that the German-language was based on the English-language gender decoder from Kat Matfield. Her work that inspired all version of gender decoders and can be found on this [GitHub profile](https://github.com/lovedaybrooke/gender-decoder) and this [website](http://gender-decoder.katmatfield.com).


# Conda Environment
```
conda create --name jobads python=3.7
conda install -n jobads ipytest==6.1.1
conda install -n jobads ipython
conda install -n jobads scikit-learn
conda install -n jobads nltk
conda install -n jobads bs4
conda install -n jobads spacy
conda install -n jobads -c conda-forge wordcloud
conda install -c conda-forge xgboost 
pip install streamlit 
python -m spacy download de_core_news_sm
```

