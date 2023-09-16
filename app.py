#Main
import streamlit as st


st.set_page_config(page_title="Resume Reviewer", page_icon="📖")


with st.sidebar:
    st.title("Resume Reviewer")
    st.markdown('''
    ## About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

 
    ''')
    st.write('Made by Spanish Indian Inquision')

