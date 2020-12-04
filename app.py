## Dependencies

# Images
from PIL import Image

# Basic Packages
import streamlit as st
import pandas as pd

# nltk Packages
import nltk
nltk.download('punkt')

# spaCy Packages
import spacy_streamlit
import spacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# Text preprocessing
import re
import unidecode
import string

# spaCy Stopwords
from spacy.lang.en.stop_words import STOP_WORDS

# Gensim Packages
from gensim.summarization import summarize
from gensim.summarization import keywords

# Sumy Summary Packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Wordcloud
from wordcloud import WordCloud

# Matplotlib Packages
import matplotlib.pyplot as plt 

## Select which english stopwords we don't want as stopwords
stopwords = list(STOP_WORDS)
deselect_words = ['no', 'not', 'never', "n't"]

for word in deselect_words:
    stopwords.remove(word)
    
## Function to get rid of accented characters
def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

## Function to remove punctuation
def remove_punctuation(text): 
    no_punct = "".join([c for c in text if c not in string.punctuation]) 
    return no_punct 

## Functions for Sumy Summarization
def sumy_luhn_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    luhn_summarizer = LuhnSummarizer()
    luhn_summarizer = LuhnSummarizer(Stemmer("english"))
    luhn_summarizer.stop_words = get_stop_words("english")
    #Summarize the document with 2 sentences
    summary = luhn_summarizer(parser.document, 2)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def sumy_lex_rank_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    lex_summarizer = LexRankSummarizer(Stemmer("english"))
    lex_summarizer.stop_words = get_stop_words("english")
    #Summarize the document with 2 sentences
    summary = lex_summarizer(parser.document, 2)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def sumy_lsa_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lsa_summarizer = LsaSummarizer()
    lsa_summarizer = LsaSummarizer(Stemmer("english"))
    lsa_summarizer.stop_words = get_stop_words("english")
    #Summarize the document with 2 sentences
    summary = lsa_summarizer(parser.document, 2)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def sumy_tr_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    tr_summarizer = TextRankSummarizer()
    tr_summarizer = TextRankSummarizer(Stemmer("english"))
    tr_summarizer.stop_words = get_stop_words("english")
    #Summarize the document with 2 sentences
    summary = tr_summarizer(parser.document, 2)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def main():
    """spaCy NLP Word Frequency App"""

    st.title("Natural Language Processing with spaCy")
    
    img = Image.open("letters.jpg")
    st.sidebar.image(img, width=300, caption = 'Image credit: towardsdatascience.com')
    menu = ["Tokenization & Word Count", "Word Similarity Check", "Extractive Text Summarization"]
    choice = st.sidebar.selectbox("Select from below", menu)

    # -------------------------------------- Tokenization & Word Count ---------------------------------------

    if choice == "Tokenization & Word Count":
        st.subheader("Tokenization & Word Count")
        menu = ["Tokenization", "Word Count", "Named Entity Recognition"]
        choice = st.selectbox("Select", menu)


    # ---------------------------------------------- Tokenization --------------------------------------------

        if choice == "Tokenization":
            st.subheader("Tokenization")
            raw_text = st.text_area("Your Text","Enter Text Here")
            nlp = spacy.load('en_core_web_sm')
            docx = nlp(raw_text)
            if st.button("Tokenize"):
                att = ['text', 'lemma_', 'pos_', 'ent_type_']
                spacy_streamlit.visualize_tokens(docx, attrs=att)


    # ----------------------------------------------- Word Count ---------------------------------------------

        elif choice == "Word Count":
            raw_text = st.text_area("Your Text","Enter Text Here")

            # Text Preprocessing
            
            raw_text = remove_punctuation(raw_text)
            raw_text = remove_accented_chars(raw_text)
            
            for i in range(0, len(raw_text)):
                raw_text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)
                raw_text = ' '.join(raw_text.split())
    
            raw_text = raw_text.lower()

            # Create a frequency table of words
            nlp = spacy.load('en_core_web_sm')

            # Build an NLP Object
            nlp = spacy.load('en_core_web_sm')
            docx = nlp(raw_text)

            # Build Word Frequency
            # word.text is tokenization in spacy

            word_frequencies = {}
            for word in docx:
                if word.text not in stopwords:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
                    
                    a = word_frequencies.keys()
                    b = word_frequencies.values()

            word_count = pd.DataFrame(list(zip(a, b)), columns =['Words', 'Frequency'])
            word_count = word_count.sort_values('Frequency', ascending  = False)

            word_count = word_count.set_index('Words')
            word_count['Frequency_%'] = word_count['Frequency']/ word_count['Frequency'].max()

            if st.button("Word Count"):
                st.write(word_count)


        # ------------------------------------- Named Entity Recognition -------------------------------------

        elif choice == "Entity Recognition":
            st.subheader("Entity Recognition")
            raw_text = st.text_area("Your Text","Enter Text Here")
            nlp = spacy.load('en_core_web_sm')
            docx = nlp(raw_text)
            if st.button("Entity Type"):
                spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

        # ----------------------------------------------------------------------------------------------------
        # ------------------------------------- End of Word Count --------------------------------------------
        # ---------------------------------------------------------------------------------------------------- 


    if choice == "Word Similarity Check":
        st.subheader("Word Similarity Check")
        raw_text1 = st.text_area("First Text","Paste/write your text here..")
        raw_text2 = st.text_area("Second Text","Another chunk of text here..")
            
        def text_prep(raw_text):        
            raw_text = remove_punctuation(raw_text)
            raw_text = remove_accented_chars(raw_text)

            for i in range(0, len(raw_text)):
                raw_text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)
                raw_text = ' '.join(raw_text.split())
    
            raw_text = raw_text.lower()

            return raw_text

        raw_text1 = text_prep(raw_text1)
        raw_text2 = text_prep(raw_text2)

        # Create a frequency table of words
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) 

        # Build an NLP Object
        nlp = spacy.load('en_core_web_sm')
        docx1 = nlp(raw_text1)
        docx2 = nlp(raw_text2)
        
        docx1_lem = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in docx1]
        docx2_lem = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in docx2]
        #word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text

        # Build Word Frequency
        ## Approximate Match
        
        def word_freq_(docx):    
            word_frequencies = {}
            for word in docx:
                if word.text not in stopwords:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1

            return word_frequencies
        
        word1 = word_freq_(docx1).keys()
        frequency1 = word_freq_(docx1).values()

        word2 = word_freq_(docx2).keys()
        frequency2 = word_freq_(docx2).values()
        
        matched_words = [x for x in word1 if x in word2]
        df = pd.DataFrame()
        df['Matched_Words'] = matched_words
    
        word_count1 = pd.DataFrame(list(zip(word1, frequency1)), columns =['Matched_Words', 'Frequency_doc1'])
        word_count2 = pd.DataFrame(list(zip(word2, frequency2)), columns =['Matched_Words', 'Frequency_doc2'])
    
        merged_df = pd.merge(df, word_count1 , on="Matched_Words")
        df = pd.merge(merged_df, word_count2, on="Matched_Words")
        
        ## Exact Match
        # word.text is tokenization in spacy

        def word_freq(docx):    
            word_frequencies = {}
            for word in docx:
                if word not in stopwords:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

            return word_frequencies
        word1 = word_freq(docx1_lem).keys()
        frequency1 = word_freq(docx1_lem).values()

        word2 = word_freq(docx2_lem).keys()
        frequency2 = word_freq(docx2_lem).values()
        
        matched_words = [x for x in word1 if x in word2]
        df_lem = pd.DataFrame()
        df_lem['Matched_Words'] = matched_words
    
        word_count1 = pd.DataFrame(list(zip(word1, frequency1)), columns =['Matched_Words', 'Frequency_doc1'])
        word_count2 = pd.DataFrame(list(zip(word2, frequency2)), columns =['Matched_Words', 'Frequency_doc2'])
    
        merged_df_lem = pd.merge(df_lem, word_count1 , on="Matched_Words")
        df_lem = pd.merge(merged_df_lem, word_count2, on="Matched_Words")

        ## What percentage matches in both docs?
        text = [raw_text1, raw_text2]

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text)

        from sklearn.metrics.pairwise import cosine_similarity
        matchpct = round(cosine_similarity(count_matrix)[0][1]*100, 2)

        if st.button('Matched Words'):
            st.write('The two documents have a', matchpct,'% match. (cosine similaity)' )
            st.subheader("Preprocessed Text")
            st.write(df)
            st.subheader("Preprocessed and Lemmatized Text")
            st.write(df_lem)  

    # ------------------------------------------------- Summarization ----------------------------------------------------
            
    elif choice == "Extractive Text Summarization":
            st.subheader("Summarize Document")
            raw_raw_text = st.text_area("Your Text","Enter Text Here")
            raw_text = raw_raw_text
            summarizer_type = st.selectbox("Select a Summarizer", ["Gensim", "Sumy Lex Rank", "Sumy Luhn", "Sumy Latent Semantic Analysis", "Sumy Text Rank"])
            if st.button('Summarize'):
                if summarizer_type == "Gensim":
                    summary_result = summarize(raw_text)
                    st.subheader("Keywords")
                    keyword = st.number_input("Enter the number of keywords and hit the 'Summarize' button.")                                
                    kw = keywords(raw_text, words = keyword).split('\n')
                    st.write(kw)
                elif summarizer_type == "Sumy Lex Rank":
                    summary_result = sumy_lex_rank_summarizer(raw_text)
                elif summarizer_type == "Sumy Luhn":
                    summary_result = sumy_luhn_summarizer(raw_text)
                elif summarizer_type == "Sumy Latent Semantic Analysis":
                    summary_result = sumy_lsa_summarizer(raw_text)
                elif summarizer_type == "Sumy Text Rank":
                    summary_result = sumy_tr_summarizer(raw_text)
                
                #------------------------------------- Length and Reading Time -------------------------------------------
                
                # Length of Original Text
                len_raw = len(raw_raw_text)

                # Length of Summary
                len_sum = len(summary_result)

                # Reading Time
                def readingtime(docs):
                    nlp = spacy.load('en_core_web_sm')
                    total_words_tokens =  [token.text for token in nlp(docs)]
                    estimatedtime  = len(total_words_tokens)/200
                    return '{} mins'.format(round(estimatedtime))

                # Reading time of the orginal document
                rt_raw = readingtime(raw_raw_text)

                # Reading time of the summary
                rt_sum = readingtime(summary_result)

                st.subheader("Summary")
                st.write(summary_result)
                
                st.subheader("Some little details.")
                st.code("The length of original document:", len_raw, "characters")
                st.code("The length of the summary:", len_sum, "characters")

                st.code("Approximate required time to read original document:", rt_raw)
                st.code("Approximate required time to read the summary:", rt_sum)
                
                # WordCloud Generation
                wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)
                wc.generate(summary_result)
                plt.imshow(wc,interpolation='bilinear')
                plt.axis("off")
                st.pyplot()

if __name__ == '__main__':
    main()


