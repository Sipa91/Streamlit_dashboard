#Underpage6
import streamlit as st


def page():
    st.subheader('References')
    st.markdown("""
    [1] TU Munich (2020, Dec 18): FührMINT Gender Decoder. https://genderdecoder.wi.tum.    
    [2] Gaucher, D., Friesen, J., & Kay, A. C. (2011). Evidence that gendered wording in job 
    advertisements exists and sustains gender inequality. Journal of Personality and Social Psychology, 101(1), 109-128.  
    [3] Pietraszkiewicz, A., Formanowicz, M., Gustafsson Sendén, M., Boyd, R. L., Sikström, S., 
    & Sczesny, S. (2019). The big two dictionaries: Capturing agency and communion in natural language. European Journal of Social Psychology, 49(5), 871-887.  
    [4] LinkedIn (2019, Mar 5) Gender Insight Report.
    https://business.linkedin.com/content/dam/me/business/en-us/talent-solutions-lodestone/body/pdf/Gender-Insights-Report.pdf?utm_source=website&utm_medium=backlink
    """)


    st.subheader('Further Reading')
    st.markdown("""
    Eagly, A. H., Nater, C., Miller, D. I., Kaufmann, M., & Sczesny, S. (2019). Gender 
    stereotypes have changed: A cross-temporal meta-analysis of US public opinion polls from 1946 to 2018. American Psychologist.   
    Hentschel, T., Heilman, M. E., & Peus, C. V. (2019). The multiple dimensions of gender 
    stereotypes: A current look at men’s and women’s characterizations of others and themselves. Frontiers in Psychology, 10, 1-19.   
    Fluchtmann, J., Glenny, A. M., Harmon, N., Maibom, J. (2020). The Gender Application 
    Gap:Do men and women apply for the same jobs? https://web2.econ.ku.dk/nharmon/docs/glenny2020gender.pdf  
    """)


    st.subheader('Acknowledgement')
    st.markdown("""
    We would like to thank Prof. Dr. Claudia Peus, Regina Dutz and the entire research team at TU Munich for providing the gender word lists and the source code of the decoder.   
      
    We would also like to point out that the German-language was based on the English-language gender decoder from Kat Matfield. 
    Her work that inspired all version of gender decoders and can be found on this [GitHub profile](https://github.com/lovedaybrooke/gender-decoder)
    and this [website](http://gender-decoder.katmatfield.com).
    """)