

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras import add_vertical_space as avs
from langchain.callbacks import get_openai_callback
import base64
import os
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter


import requests
import streamlit as st 
from streamlit_lottie import st_lottie
from PIL import Image



with st.sidebar:
    st.image("panda_logo.png", caption="")
    st.title("Resume Reviewer")
    st.markdown('''
    ## About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

    
    ''')


    


image =Image.open('teampic.jpg') 

with st.container():
    left_column,right_column = st.columns(2)
    with left_column:
        st.header("Resume     Resurrection")
        st.divider()
        st.subheader("About")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
    
lottie_coding = load_lottieurl("https://lottie.host/e6ce3de4-1da6-4d60-84d2-100f97d68b37/otWBBaQhxU.json")

with right_column:
    st_lottie(lottie_coding,height =200,key="paperplane")

st.image(image, caption="team")