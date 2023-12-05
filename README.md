# Configurable Enterprise Chat Agent
This Chat Agent is build specifically as a reusable and configurable sample app to share with enterprises or prospects. 

1. It uses [LangChain](https://www.langchain.com/) as the framework to easily set up LLM Q&A chains
2. It uses [Streamlit](https://streamlit.io/) as the framework to easily create Web Applications
3. It uses [Astra DB](https://astra.datastax.com/) as the Vector Store to enable Rerieval Augmented Generation in order to provide meaningfull contextual interactions
4. It uses [Astra DB](https://astra.datastax.com/) as Short Term Memory to keep track of what was said and generated
5. It uses a StreamingCallbackHandler to stream output to the screen which prevents having to wait for the final answer
6. It allows for new Content to be uploaded, Vectorized and Stored into the Astra DB Vector Database so it can be used as Context
7. It offers a configurable localization through `localization.csv`
8. It offers a guided experience on-rails through `rails.csv`

## Preparation
1. First install the Python dependencies using:
```
pip3 install -r requirements.txt
```
2. Then update the `OpenAI`, `AstraDB` and optionally `LangSmith` secrets in `streamlit-langchain/.streamlit/secrets.toml`. There is an example provided at `secrets.toml.example`.

## Customization
Now it's time to customize the app for your specific situation or customers.
### Step 1
Define credentials by adding a new username and password in the `[passwords]` section in `streamlit-langchain/.streamlit/secrets.toml`.
### Step 2
Define the UI language of the app by adding a localization code in the `[languages]` section in `streamlit-langchain/.streamlit/secrets.toml`. Currently `en_US` and `nl_NL` are supported. However it is easy to add additional languages in `localization.csv`.
### Step 3
Create a guided experience by providing sample prompts in `rails.csv`. The convention here is that `<username>` from Step 1 is used to define the experience.
### Step 4
Start up the app and pre-load relevant PDF and Text files so that the app has content that can be used as context for the questions/prompts in the next step. All this data will be loaded into a user specific table defined by `<username>`.
### Step 5
Create a customized welcome page in the root folder. The convention here is to create a markdown file called `<username>.md`. Ideally, list which files have been pre-loaded.

## Getting started
You're ready to run the app as follows:
```
streamlit run app.py
```
In addition to the pre-loaded content, a user can add additional content that will be used as context for prompts.

## Deploy to the internet
It's easy to upload this app to the community edition of Streamlit. As the app uses a login page it is safe to have it publicly available.

## Warning
The goal of this app is to be easily shared within enterprises. Just be aware that YOUR OPENAI subscription is being used for creating embeddings and LLM calls. This WILL incur cost.