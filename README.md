# asc_rag_chatbot

The deployed app is available here: https://ascragbot-fnbmsd9sdtaqc77x75r7zi.streamlit.app/

## How the app works

This chatbot is able to answer questions about Adult Social Care webpages, with a focus on content related to Caring For Someone, Social Care and Assessments, and Independent Living.

The chatbot uses a technique called Retrieval Augmented Generation (RAG) to retrieve contextual information from the relevant webpages and makes this available to OpenAI's GPT3.5 turbo model, so that it can provide a more contextualised response to the user.

The app uses the OpenAI API to achieve this. The main application code is available in app.py, while various core functionality is defined in utils.py.

The high level logic / steps in app.py is as follows:
- Loads the embeddings cache - this actually happens in utils.py
- Loads and parses the html documents in 'downloaded_pdfs' - see the load_documents function in utils.py
- Generates embeddings for all of the webpages - see the get_embedding, embedding_from_string and get_embeddings functions in utils.py
- Generates a series of urls for the bot to include in its responses
- Builds out the streamlit front-end
- Initialises the session state for the model - this is where the model is prompted to provide it with instructions and examples of good responses
- Sets out further logic for the UI, and to allow for an ongoing conversation
- In this final section it also calls the retrieve_documents function from utils.py. This is the key one that looks at the user's prompt and then retrieves the most similar documents from downloaded_pdfs and includes these in the model's context. This is what allows the model to provide a more contextualised response.

## Data

The data that has been used can be found in the folder 'downloaded_pdfs' (these are in fact html files!). These are all webpages, primarily from from kingston.gov.uk.

The file 'asc_website.csv' provides a full list of the pages that have been used and their urls.

In addition to the two .py files, there is also a file called extract.ipynb, which can be used to extract the webpages as html documents. To add more webpages, simply add the title and urls to the 'asc_website.csv' file and re-run extract.ipynb and they will be downloaded to downloaded_pdfs.

## Evaluation

Some automated evaluation of the model's responses has been carried out in the file 'evaluate.ipynb'.

This file uses the model and actual answers in the spreadsheet 'AI assistant -  questions and answers'. It creates vector embeddings of these answers and then compares them using cosine similarity. Nearly all of the answers have a similarity of over 0.9, which shows a strong match between the model answer and the answer provided by the chatbot.

## Running the app locally

First clone this repository on to your local machine.

Next, ensure that you are running the correct version of Python, which is 3.9.7.

Then, in your terminal create a virtual environment and install the required packages:

```
python -m venv asc_rag_chatbot_venv
source asc_rag_chatbot_venv/bin/activate
pip install -r requirements.txt
```

Next, create a file called .env and put your OpenAI API key inside e.g.:

```
OPENAI_API_KEY= 'your key goes here'
```

Then create a folder called .streamlit and inside it create a file called secrets.toml. Also put your OpenAI API key inside this e.g.:

```
OPENAI_API_KEY= 'your key goes here'
```

Next, run the app using the command:

```
streamlit run app.py
```

## Deploying the app to Streamlit

Deploying the app to Streamlit is very straighforward
1. First go to platform.openai.com and create an account
2. Got to dashboard > API keys > Create secret key
3. Create a key and make a copy of it
4. Visit streamlit.io and create an account
5. Connect your Streamlit account to the relevant GitHub account that has the app code
6. Click on Create App and then 'Yup, I have an app'
7. Select the relevant repository, branch and file path
8. Select 'Advanced settings', select Python version 3.9, and in the Secrets section enter OPENAI_API_KEY= ''. Enter your OpenAI API key between the apostrophes
9. Click on 'Deploy!'
