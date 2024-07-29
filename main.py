import os
import json
from openai import OpenAI
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

import custom_tools as ct
import utils

tools_list = ct.return_tools_list()

st.title("Simple BSK chatbot")

# st.sidebar
model_option = st.sidebar.selectbox(
    "Choose LLM Model",
    ("OpenAI (API)", "Llama Groq (Local)"),
)
# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    utils.add_sys_msg(st.session_state.chat_history)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt:= st.chat_input("What is up?"):
    
    # Add user message to chat history
    utils.add_user_msg(st.session_state.chat_history, prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if model_option == "OpenAI (API)":
            with st.spinner('Thinking...'):
                response_message = utils.chat_completion(st.session_state.chat_history, tools_list)
        else:
            with st.spinner('Thinking...'):
                response_message = utils.chat_completion_ollama(st.session_state.chat_history, tools_list)
        tool_calls = response_message.tool_calls

        # tool call handling
        if tool_calls:
            st.toast('⚙️ A function was called!..')
            print('tool used')
            for tool_call in tool_calls:
                tool_id, tool_name, tool_arguments, tool_params = utils.extract_tool_details(tool_call)
                utils.add_tool_detail(st.session_state.chat_history, tool_id, tool_name, tool_arguments, tool_params)
                tool_call_result = eval(f"ct.{tool_name}")(**tool_params)
                utils.add_tool_response(st.session_state.chat_history, tool_id, tool_name, tool_call_result)
            
            if model_option == "OpenAI (API)":
                with st.spinner('Thinking...'):
                    response_message = utils.chat_completion(st.session_state.chat_history, tools_list)
            else:
                with st.spinner('Thinking...'):
                    response_message = utils.chat_completion_ollama(st.session_state.chat_history, tools_list)

            response_message = response_message.content
            st.markdown(response_message)
        else:
            response_message = response_message.content
            st.markdown(response_message)

    # Add assistant response to chat history
    utils.add_ai_msg_string(st.session_state.chat_history, response_message)
    st.session_state.messages.append({"role": "assistant", "content": response_message})