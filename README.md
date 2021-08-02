### What's done in this project?

- Tokenization: spaCy segments the text into sentences, words, and punctuation marks. These are called tokens. Texts are tokenized using spaCy model ‘en_core_web_sm’ and visualized in a table as words and punctuations, their Lemmatization, Parts of Speech, and type of the entity.

- Word Count: Preprocessed the text (removal of stop-words, punctuations, accented characters, etc.) Computed the word frequency table by computing the number of times each word is present in the document. Finally, the frequency of each word and their percentage are shown in a DataFrame.

- Matched Words from Two Documents: spaCy takes inputs from the user, cleans the texts, uses model ‘en_core_web_sm’ to find out Lemmatized words. If the same lemmatized words exist in the two documents, the app shows the words and their frequencies in each document in Tabular format. 

- Extractive Text Summarization: Five different text summarizers are made available in this app. Gensim, Sumy Lex Rank, Sumy Luhn, Sumy Latent Semantic Analysis, and Sumy Text Rank provide summarized text, time to read the original and summarized document, and simple word clouds generated from the words of the summarized document.

**Libraries and frameworks**: Pandas, Nltk, Spacy, spacy_streamlit, en_core_web_sm, Genism, Sumy, Sklearn, Wordcloud, Matplotlib, Unicode, Pillow, Streamlit, Heroku

### [Go to the App](https://laaqipm25.herokuapp.com/)

[![Watch Demo Here](https://github.com/SumaiaParveen/Regression-LA-AQI-Prediction/blob/main/AQI.JPG)](https://laaqipm25.herokuapp.com/)
