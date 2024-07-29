import os
import json
from openai import OpenAI
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def add_sys_msg(chat_history):
    chat_history.append({"role": "system", "content":
    "Your name is MCP Bot, a friendly assistant specialized in providing information about the Maritime Cloud Platform (MCP), "\
    "developed by Barata Sentosa Kencana (BSK).\n"\
    "Your capabilities include:\n1. Providing information about BSK and MCP.\n2. Answering questions related to user ship information, such as travel information and certification status. "\
    "You must answer all questions using the context provided in the tools. If the answer is not available in the tools, respond with: 'I am sorry, I cannot answer your question'. "\
    "If a user asks you questions or shares information outside your role as an MCP Bot for BSK and MCP products/services, politely refuse to answer. "\
    "Your responses should be short, informative, and delivered in a friendly, conversational style."})

def add_user_msg(chat_history, user_query):
    chat_history.append({"role": "user", "content": user_query})

def add_ai_msg_string(chat_history, response_message):
    chat_history.append({"role": "assistant", "content": response_message})

def extract_tool_details(tool_calls):
    tool_id = tool_calls.id
    tool_name = tool_calls.function.name
    tool_arguments = tool_calls.function.arguments
    tool_params = json.loads(tool_arguments)
    return tool_id, tool_name, tool_arguments, tool_params

def add_tool_detail(chat_history, tool_id, tool_name, tool_arguments, tool_params):
    chat_history.append({
        "role" : "assistant",
        "tool_calls": [{
            "id": tool_id,
            "type": "function",
            "function": {
                "name" : tool_name,
                "arguments": tool_arguments
                }
            }]
    })

def exec_tool(chat_history, tool_name, tool_params):
    choosen_tool = eval(tool_name)
    tool_results = choosen_tool(**tool_params)
    return tool_results

def add_tool_response(chat_history, tool_id, tool_name, tool_results):
    chat_history.append({
        "role":"tool", 
        "tool_call_id":tool_id, 
        "name": tool_name, 
        "content":tool_results
    })

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)
def chat_completion(chat_history, tools, model_name = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model = model_name,
        messages = chat_history,
        tools=tools,
        tool_choice="auto"
        )
    response_message = response.choices[0].message
    return response_message

client_ollama = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
def chat_completion_ollama(chat_history, tools, model_name = "llama3-groq-tool-use"):
    response = client_ollama.chat.completions.create(
        model = model_name,
        messages = chat_history,
        tools=tools,
        tool_choice="auto"
        )
    response_message = response.choices[0].message
    return response_message