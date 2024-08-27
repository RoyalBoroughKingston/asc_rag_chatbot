#Load libraries
from utils import *
import streamlit as st
from openai import OpenAI
import pandas as pd

#Load the environment variables from the st.secrets file
key = st.secrets["OPENAI_API_KEY"]

#Load and pre-process documents
texts = load_documents('downloaded_pdfs')

#Generate embeddings for documents
embeddings = get_embeddings(texts, model="text-embedding-3-small")

#Initialize OpenAI client
client = OpenAI(api_key=key)

#Function to generate a response from OpenAI
def generate_response(messages):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return completion.choices[0].message.content

#Generate links string
links = pd.read_csv('asc_website.csv', encoding='cp1252')

#Initialize the starting part of the string
links_prompt_string = "Make sure that you include links in your responses to the most relevant pages on the website for the user's query. This can include for further information, as well as to take next steps. A list of pages and links is available here: "

# Iterate over each row in the DataFrame and append to the string
#for index, row in links.iterrows():
    #links_prompt_string += f"\n- {row['title']}: {row['url']}"

for index, row in links.iterrows():
    links_prompt_string += f"\n- [{row['title']}]({row['url']})"

links_prompt_string = links_prompt_string.replace('[', '').replace(']', '')

#Streamlit app
st.title("ASC Carers Information AI Assistant")

st.write("""
This chatbot can provide you with information about being a carer in Kingston. 
Simply enter a query and press 'send' to start the conversation.
""")

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = [
    
    #Provide information about the task, the role the LLM should assume and the expected output
    {"role": "system", "content": "Your role is to provide advice to members of the public who live in Kingston and are looking for information about local social care services and being a carer. You will receive a prompt from the user and some relevant context from Kingston's website pages. You should use these to provide an accurate response to the user's question and provide links to where they can find more information and/or complete next steps"},

    {"role": "system", "content": links_prompt_string},

    {"role": "system", "content": "Also remember to break up your response into paragraphs, use British English (for exammple 'day centres' instead of 'data centers' and 'personalised' instead of 'personalized') and to encourage users to self-serve (for example instead or providing an email or contact number, direct them towards forms they can complete themselves)"},

    ####Few-shot examples

    #Provide information about the task, the role the LLM should assume and the expected output
    {"role": "system", "content": "Here are some examples of possible queries from a user and a good response to that query"},

    #Provide prompt
    {"role": "user", "content": "Can my mum get care from the council?", "example": True},
    
    #Provide expected output
    {"role": "assistant", "content": "Your mum may be able to get care from the council, but it depends on her specific needs and financial situation. We will carry out an assessment to determine her eligibility for care services. This assessment will look at her care needs and financial circumstances. If your mum meets the criteria for care and has less than £23,250 in savings and assets, she could qualify for council-funded care. She might need to pay for some or all of her care if she has more than this amount. For more information, you can explore the following resources: Eligibility for care services: https://www.kingston.gov.uk/eligiblecareservices and About care needs assessments: https://www.kingston.gov.uk/request-assessment", "example": True},

    #Provide prompt
    {"role": "user", "content": "How do I get a care assessment?", "example": True},
    
    #Provide expected output
    {"role": "assistant", "content": "You can explore what care you might need by completing the care needs checklist. This checklist will help you understand if you're likely to qualify for support from the council and will guide you to local support and the form to request a care needs assessment. Once you submit the request for a care needs assessment, the process typically begins with a phone conversation to discuss your needs. If we agree together to move forward with the assessment, a member of the adult social care team will contact you within 7 days to schedule an appointment. The assessment itself is free, and you will receive the outcomes in 6 weeks. If you qualify for council-funded care, there may be costs, including savings and assets, based on her financial situation. You can also explore other help available through community partners via Connected Kingston. You can start the care needs checklist for more details or to start the process: https://find-care-support-adults.kingston.gov.uk/", "example": True},

    #Provide prompt
    {"role": "user", "content": "Can the council supply a stairlift?", "example": True},
    
    #Provide expected output
    {"role": "assistant", "content": "The council can help you get a stairlift. If you need a stairlift or other major adaptations to your home, you should first get an occupational therapy assessment. This will determine if such adaptations are suitable for you. To arrange an assessment, you can contact the occupational therapy team by calling 020 8547 5005, Monday to Friday from 9am to 4.45pm. Alternatively, you can ask your GP or hospital to refer you. If you qualify for a stairlift, you may also be eligible for a Disabled Facilities Grant: https://www.gov.uk/disabled-facilities-grants to help cover the costs.", "example": True},
  
  ]

# Display conversation history, excluding initial setup messages
for message in st.session_state.messages:
    if message['role'] == 'user' and 'example' not in message:
        st.text_area("You:", message['content'], key=f"user_{message['content']}", height=100, max_chars=2000)
    elif message['role'] == 'assistant' and 'example' not in message:
        st.markdown(f"**Assistant:** {message['content']}")

# Input form to avoid duplicating text_input key
with st.form(key='user_input_form', clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    # Add user message to conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve relevant text
    retrieved_text = retrieve_documents(user_input, embeddings, texts, model="text-embedding-3-small")

    # Add retrieved text to the context
    context = f"Context: {retrieved_text}\n\nUser Query: {user_input}"

    # Generate response from OpenAI
    response = generate_response(st.session_state.messages + [{"role": "system", "content": context}])

    # Add assistant message to conversation history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # To clear the text input after submission
    st.experimental_rerun()

