import streamlit as st

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

import tempfile
import os
import pandas as pd

from langchain.document_loaders import PyPDFLoader

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#####################
### Session Cache ###
#####################

# Cache Chat Memory for future runs
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        return_messages=True,
        k=top_k
    )
memory = st.session_state.memory

######################
### Resource Cache ###
######################

# Cache Astra DB session for future runs
@st.cache_resource(show_spinner="Setting up Astra DB connection...")
def load_session():
    # Connect to Astra DB
    cluster = Cluster(cloud={'secure_connect_bundle': st.secrets["ASTRA_SCB_PATH"]}, 
                      auth_provider=PlainTextAuthProvider(st.secrets["ASTRA_CLIENT_ID"], 
                                                          st.secrets["ASTRA_CLIENT_SECRET"]))
    return cluster.connect()
session = load_session()

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner="Getting the OpenAI embedding...")
def load_embedding():
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
embedding = load_embedding()

# Cache Vector Store for future runs
@st.cache_resource(show_spinner="Getting the Vector Store from Astra DB...")
def load_vectorstore():
    # Get the vector store from Astra DB
    return Cassandra(
        embedding=embedding,
        session=session,
        keyspace='vector_preview',
        table_name='vector_context'
    )
vectorstore = load_vectorstore()

# Cache OpenAI Chat Model for future runs
@st.cache_resource(show_spinner="Getting the OpenAI Chat Model...")
def load_model():
    # Get the OpenAI Chat Model
    return ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        verbose=False)
llm = load_model()

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file):
    if uploaded_file is not None:
        docs = []

        if uploaded_file.name.endswith('txt'):
            file = [uploaded_file.read().decode()]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 200
            )  
            texts = text_splitter.create_documents(file)
            vectorstore.add_documents(texts)  
            st.info(f"{len(texts)} chuncks loaded into Astra DB")

        if uploaded_file.name.endswith('pdf'):
            
            # Read PDF
            docs = []
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print("""Processing: {file}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 200
            )  
            pages = text_splitter.split_documents(docs)
            vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} pages loaded into Astra DB")

# Cache Conversational Chain for future runs
@st.cache_resource(show_spinner="Getting the Conversational Chain...")
def load_qa_chain():

    template = """
You're a helpful AI assistent tasked to answer the user's question.
You're friendly and you answer extensively with multiple sentences and preferably use bullets.
If you don't know the answer, just say so.

Use the following context to answer the question:
{context}

Use the following history to answer the question:
{chat_history}

Answer this question:
{question}
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"], 
        template=template
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False,
        verbose=False,
        memory=ConversationBufferWindowMemory(memory_key="chat_history", 
                                        input_key='question', 
                                        output_key='answer',
                                        return_messages=True,
                                        k=3),
        combine_docs_chain_kwargs={"prompt": prompt}, 
        chain_type="stuff"
    )
qa_chain = load_qa_chain()

# Drop previously existing vector data
def drop_vector_data():
    session.execute(f"DROP TABLE IF EXISTS vector_preview.vector_context")

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader("Upload a document for additional context", type=['txt', 'pdf'], )
        submitted = st.form_submit_button("Save")
        if submitted:
            vectorize_text(uploaded_file)

# Drop the vector data and start from scratch
with st.sidebar:
    with st.form('drop'):
        st.caption(lang_dict['drop_context'])
        submitted = st.form_submit_button("Delete context")
        if submitted:
            with st.spinner("Removing context and intialising..."):
                #drop_vector_data()
                vectorstore.clear()
                vectorstore = load_vectorstore()

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="Hi, I'm your personal assistant and am ready to help!")]

# Redraw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.role).markdown(message.content)

# Now get a prompt from a user
if prompt := st.chat_input("What's up?"):
     # Add the prompt to messages, stored in session state
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))

    # Draw the prompt on the page
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the results from Langchain
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        print(f"Getting LLM response for: {prompt}")
        print(f"Using memory: {history}")
        callback = StreamHandler(response_placeholder)
        response = qa_chain.run({'question': prompt, 'history': history}, callbacks=[callback])
        print(f"Response: {response}")
        
        # Write the final answer without the cursor
        response_placeholder.markdown(response)

        # Add the answer to the messages session state
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))