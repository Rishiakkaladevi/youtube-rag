from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled, _errors
from youtube_transcript_api.proxies import GenericProxyConfig


from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import requests
import os
from functools import partial
load_dotenv()
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")  # Put your key in .env
if not SCRAPERAPI_KEY:
    raise ValueError("SCRAPERAPI_KEY environment variable is not set")

# Configure proxy for ScraperAPI
proxy = {
    "http": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001",
    "https": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001",
}
cookies = {"CONSENT": "YES+1"}

# Monkey patch requests.get to always use proxy + cookie (retained from original)
requests.get = partial(requests.get, proxies=proxy, cookies=cookies, timeout=10)

# Configure YouTubeTranscriptApi with proxied requests
proxy_config = GenericProxyConfig(
    http_url=f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001",
    https_url=f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"
)
ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config, cookies=cookies)

def mainfun(id,question):
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = ' '.join([x['text'] for x in transcript_data])
    except _errors.TranscriptsDisabled:
        return "Captions are disabled for this video."
    except _errors.VideoUnavailable:
        return "The video is unavailable."
    except _errors.IpBlocked:
        return "Your IP (or proxy) is blocked by YouTube. Try again later."
    except Exception as e:
        return f"Error: {str(e)}"


    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
    chunks = text_splitter.create_documents([transcript])


    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks,embedding_model)

    #vectorstore.index_to_docstore_id

    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={'k':4})

    load_dotenv()
    llm=HuggingFaceEndpoint(
        repo_id='moonshotai/Kimi-K2-Instruct',
        task = 'text-generation'
        )
    model = ChatHuggingFace(llm=llm)

    prompt= PromptTemplate(
        template=''' You are a helpful assistant.
                Answer ONLY from the provided transcript content.
                If the context is insufficient,say that you don't know
                {context}
                question : {question}''',
                input_variables=['context','question']
                )

    question = " "
    retrieved_docs = retriever.invoke(question)

    def contextret(retrieved_docs):
        contextpara = '\n\n'.join(content.page_content for content in retrieved_docs)
        return contextpara

    #final_prompt=prompt.invoke({"context": contextret(retrieved_docs) ,"question":question})


    parallelchain = RunnableParallel({
        "context" : retriever | RunnableLambda(contextret),
        "question" : RunnablePassthrough()
    })

    parser = StrOutputParser()

    mainchain = parallelchain | prompt | model | parser




    return mainchain.invoke(question)


st.title("Youtube Video Bot")
video_id= st.text_input(label="",placeholder="enter your youtube video id here ")
if st.button("submit"):
    st.session_state.submitted=True
if st.session_state.get("submitted",False):
    query=st.text_input(label="Ask me about the video")
    if st.button("answer"):
        st.text(mainfun(video_id,query))
        

