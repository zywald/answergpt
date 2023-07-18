import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from openai.error import OpenAIError

st.title("🦜🔗  - AnswerGPT")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Selection between two types of messages
message_type = st.sidebar.selectbox(
    "Select the type of message:", options=['Chat', 'Email'])

# Selection between 4 tones
tone = st.sidebar.selectbox("Select the tone of the message:", options=[
                            'Professional', 'Friendly', 'Casual', 'Formal'])

# Selection between two models
model_name = st.sidebar.selectbox("Select the model:", options=['gpt-3.5-turbo', 'gpt-4'],
                                  help="Choose 'gpt-3.5-turbo' for fast answer, or 'gpt-4' for more complex messages.")


def answer_message(original_message, message_to_reformulate):
    try:
        # Instantiate LLM model
        llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key)

        # Define common instructions
        common_instructions = (
            "You are an expert assistant trained to write clear, concise, and well-structured messages. "
            "For email messages, use formatting techniques like bolding, italics, bullet points, and paragraph breaks where appropriate. "
            "Your tone should be a balanced mix of conversational and Spartan efficiency. "
            "For chat messages, do not sign the sender's name at the end. "
            "You will answer in the email to reply and the draft email or indication language. "
        )

        # Define specific instructions based on user input
        specific_instructions = f"Write a {{message_type}} message in a {{tone}} tone. "
        if original_message:
            specific_instructions += f"The original message to reply to is: '{original_message}'. "
        if message_to_reformulate:
            specific_instructions += f"Reformulate this draft or use the following indications for you to craft the message: '{message_to_reformulate}'. "
        else:
            specific_instructions += "Provide a potential reply."

        # Combine common and specific instructions into the prompt
        template = common_instructions + specific_instructions

        prompt = PromptTemplate(
            input_variables=["message_type", "tone"], template=template)
        prompt_query = prompt.format(message_type=message_type, tone=tone)

        # Run LLM model
        with st.spinner('Generating response...'):
            response = llm(prompt_query)

        # Print results
        st.write(response)
    except OpenAIError as e:
        st.error(f"An OpenAI API error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unknown error occurred: {str(e)}")


with st.form("myform"):
    original_message = st.text_area("Original Message to answer (optional):",
                                    help="Enter the original message that you want to reply to.")
    message_to_reformulate = st.text_area(
        "Message Draft to formulate:", help="Enter a draft of the message that you want to reformulate.")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif not original_message and not message_to_reformulate:
        st.info("Please enter an original message or a message draft.")
    elif submitted:
        answer_message(original_message, message_to_reformulate)
