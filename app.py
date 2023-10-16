import pandas as pd 
import pickle as pkl
import streamlit as st
import numpy as np

st.title(':blue[COMMENT CLASSIFIER] :face_with_monocle:')

#model
rf_clf=pkl.load(open('model.pkl','rb'))

#vertorizar
vetorizar=pkl.load(open('vectorizar.pkl','rb'))

#user id input
comment=st.text_input('Enter ',placeholder='paste comment here')



def cc(predicted_sentences):
  here=predicted_sentences[0]
  mapping = {'TOXIC :rage:': 1, 'SEVERE TOCIX :imp:': 1 , 'OBSCENE :o2: ': 1 , 'THREAT :bomb:': 1, 'INSULT 	:man-gesturing-no:': 1 , 'IDENTITY HATE :busts_in_silhouette:': 1}
  input_list = list(here)
  list1 =[key for i, key in enumerate(mapping.keys()) if input_list[i]==1]
  return list1


if st.button('Review'):
    if comment:
      new_sentences = list(comment)
      new_sentence_tfidf = vetorizar.transform(new_sentences)
      predicted_sentences = rf_clf.predict(new_sentence_tfidf)
      l=cc(predicted_sentences)

      st.caption('COMMENT IS BEING CLASSIFIED AS')
      for i in l:
         st.markdown("- " + i)

