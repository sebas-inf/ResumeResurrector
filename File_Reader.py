import streamlit as st
import openai
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import my_key



def read_resume(jd, resume, options):
    df = analyze_str(resume, options)
    df_string = df.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x).to_string(index=False)
    st.write("OpenAI comprehensive analysis..")
    summary_question = f"Job requirements: {{{jd}}}" + f"Resume summary: {{{df_string}}}" + ", please directly provide the candidate's match summary for this position (within 200 words);'"
    summary = ask_openAI(summary_question)
    df.loc[len(df)] = ['Comprehensive summary', summary]
    extra_info = "Scoring requirements: top 10 universities in China +3 points, 985 universities +2 points, 211 universities +1 point, top company experience +2 points, well-known company +1 point, overseas background +3 points, foreign company background +1 point. "
    score_question = f"Job requirements: {{{jd}}}" + f"Resume summary: {{{df.to_string(index=False)}}}" + ", please directly return the candidate's match score for this job (0-100), please score accurately for easy comparison and ranking for other candidates, '" + extra_info
    score = ask_openAI(score_question)
    df.loc[len(df)] = ['Match score', score]

    return resume

def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(resume)

    embeddings = OpenAIEmbeddings(openai_api_key=my_key.get_key())
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    df_data = [{'option': option, 'value': []} for option in options]
    st.write("Information Retrieval")

    # Create progress bar and empty element
    progress_bar = st.progress(0)
    option_status = st.empty()

    for i, option in tqdm(enumerate(options), desc="Information Retrieval in progress", unit="option", ncols=100):
        question = f"What is the {option} of this candidate, please return a concise answer, up to 250 words, if not found, return 'Not provided'"
        docs = knowledge_base.similarity_search(question)
        llm = OpenAI(openai_api_key=my_key.get_key(), temperature=0.3, model_name="text-davinci-003", max_tokens="2000")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question )
        df_data[i]['value'] = response
        option_status.text(f"Fetching information for: {option}")

        # Update progress bar
        progress = (i + 1) / len(options)
        progress_bar.progress(progress)

    df = pd.DataFrame(df_data)
    st.success("Resume elements retrieved")
    return df