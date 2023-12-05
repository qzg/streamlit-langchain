import os
from pathlib import Path
import hmac
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ["LANGCHAIN_PROJECT"] = st.secrets['LANGCHAIN_PROJECT']

import pandas as pd

from astrapy.db import AstraDB

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import AstraDB
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import AstraDBChatMessageHistory

import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.schema import HumanMessage, AIMessage, Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

from langchain.callbacks.base import BaseCallbackHandler

print("Started")

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 4
top_k_memory = 3

###############
### Globals ###
###############

global lang_dict
global rails_dict
global embedding
global vectorstore
global retriever
global model
global chat_history
global memory

#################
### Functions ###
#################

# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(st.session_state['password'], st.secrets.passwords[st.session_state['username']]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    # Return True if the username + password is validated.
    if st.session_state.get('password_correct', False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 User not known or password incorrect')
    return False

def logout():
    del st.session_state.password_correct
    del st.session_state.user

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(f"""Processing: {file}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())

            # Create the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 100
            )

            if uploaded_file.name.endswith('txt'):
                file = [uploaded_file.read().decode()]
                texts = text_splitter.create_documents(file, [{'source': uploaded_file.name}])
                vectorstore.add_documents(texts)
                st.info(f"{len(texts)} {lang_dict['load_text']}")

            if uploaded_file.name.endswith('pdf'):
                # Read PDF
                docs = []
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())

                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)  
                st.info(f"{len(pages)} {lang_dict['load_pdf']}")

##################
### Data Cache ###
##################

# Cache localized strings
@st.cache_data()
def load_localization(locale):
    print("load_localization")
    # Load in the text bundle and filter by language locale
    df = pd.read_csv("localization.csv")
    df = df.query(f"locale == '{locale}'")
    # Create and return a dictionary of key/values.
    lang_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return lang_dict

# Cache localized strings
@st.cache_data()
def load_rails(username):
    print("load_rails")
    # Load in the rails bundle and filter by username
    df = pd.read_csv("rails.csv")
    df = df.query(f"username == '{username}'")
    # Create and return a dictionary of key/values.
    rails_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return rails_dict

#############
### Login ###
#############

# Check for username/password and set the username accordingly
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

username = st.session_state.user
language = st.secrets.languages[username]
lang_dict = load_localization(language)

#######################
### Resources Cache ###
#######################

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner=lang_dict['load_embedding'])
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    return OpenAIEmbeddings()

# Cache Vector Store for future runs
@st.cache_resource(show_spinner=lang_dict['load_vectorstore'])
def load_vectorstore(username):
    print("load_vectorstore")
    # Get the load_vectorstore store from Astra DB
    return AstraDB(
        embedding=embedding,
        collection_name=f"vector_context_{username}",
        token=st.secrets["ASTRA_VECTOR_TOKEN"],
        api_endpoint=os.environ["ASTRA_VECTOR_ENDPOINT"],
    )
    
# Cache Retriever for future runs
@st.cache_resource(show_spinner=lang_dict['load_retriever'])
def load_retriever():
    print("load_retriever")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

# Cache OpenAI Chat Model for future runs
@st.cache_resource(show_spinner=lang_dict['load_model'])
def load_model():
    print("load_model")
    # Get the OpenAI Chat Model
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4-1106-preview',
        streaming=True,
        verbose=True
    )

# Cache Chat History for future runs
@st.cache_resource(show_spinner=lang_dict['load_message_history'])
def load_chat_history(username):
    print("load_chat_history")
    return AstraDBChatMessageHistory(
        session_id=username,
        api_endpoint=os.environ["ASTRA_VECTOR_ENDPOINT"],
        token=st.secrets["ASTRA_VECTOR_TOKEN"],
    )

@st.cache_resource(show_spinner=lang_dict['load_message_history'])
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

# Cache prompt
@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in the user's language:"""

    return ChatPromptTemplate.from_messages([("system", template)])

#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

############
### Main ###
############

# Write the welcome text
try:
    st.markdown(Path(f"""{username}.md""").read_text())
except:
    st.markdown(Path('welcome.md').read_text())

# DataStax logo
with st.sidebar:
    st.image('./assets/datastax-logo.svg')
    st.text('')

# Logout button
with st.sidebar:
    with st.form('logout'):
        st.caption(f"""{lang_dict['logout_caption']} '{username}'""")
        st.form_submit_button(lang_dict['logout_button'], on_click=logout)

# Initialize
with st.sidebar:
    rails_dict = load_rails(username)
    embedding = load_embedding()
    vectorstore = load_vectorstore(username)
    retriever = load_retriever()
    model = load_model()
    chat_history = load_chat_history(username)
    memory = load_memory()
    prompt = load_prompt()

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader(lang_dict['load_context'], type=['txt', 'pdf'], accept_multiple_files=True)
        submitted = st.form_submit_button(lang_dict['load_context_button'])
        if submitted:
            vectorize_text(uploaded_file)

# Drop the Conversational Memory
with st.sidebar:
    with st.form('delete_memory'):
        st.caption(lang_dict['delete_memory'])
        submitted = st.form_submit_button(lang_dict['delete_memory_button'])
        if submitted:
            with st.spinner(lang_dict['deleting_memory']):
                memory.clear()

# Drop the vector data and start from scratch
if (username in st.secrets['delete_option'] and st.secrets.delete_option[username] == 'True'):
    with st.sidebar:
        with st.form('delete_context'):
            st.caption(lang_dict['delete_context'])
            submitted = st.form_submit_button(lang_dict['delete_context_button'])
            if submitted:
                with st.spinner(lang_dict['deleting_context']):
                    vectorstore.clear()
                    memory.clear()
                    st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

# Draw rails
with st.sidebar:
        st.subheader(lang_dict['rails_1'])
        st.caption(lang_dict['rails_2'])
        for i in rails_dict:
            st.markdown(f"{i}. {rails_dict[i]}")

# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
if question := st.chat_input(lang_dict['assistant_question']):
    print(f"Got question {question}")

    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print(f"Draw prompt")
    with st.chat_message('human'):
        st.markdown(question)

    # Get the results from Langchain
    print(f"Chat message")
    with st.chat_message('assistant'):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        print(f"Using memory: {history}")

        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        })
        print(f"Using inputs: {inputs}")

        chain = inputs | prompt | model
        print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)]})
        print(f"Response: {response}")
        print(embedding.embed_query(question))
        content = response.content

        # Write the sources used
        relevant_documents = retriever.get_relevant_documents(question)
        content += f"""
        
*{lang_dict['sources_used']}:*  
"""
        sources = []
        for doc in relevant_documents:
            source = doc.metadata['source']
            page_content = doc.page_content
            if source not in sources:
                content += f"""📙 :orange[{os.path.basename(os.path.normpath(source))}]  
"""
                sources.append(source)
        print(f"Used sources: {sources}")

        # Write the final answer without the cursor
        response_placeholder.markdown(content)

        # Add the result to memory
        memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))

with st.sidebar:
            st.caption("v11.20.01")
