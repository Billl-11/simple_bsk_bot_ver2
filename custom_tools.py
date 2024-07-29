import random
import datetime
import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

def return_tools_list():
    tools_list = [
        {   "type": "function",
            "function": {
                "name": "get_ship_travel_info",
                "description": "Tools provided by MCP to get travel information of the desired ship, including origin port/harbor, " \
                    "destination port/harbor, and estimated time of arrival",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ship_name": {
                            "type": "string",
                            "description": "The name of the ship for which the travel information is being checked",
                        },
                    },
                    "required": ["ship_name"],
                },
            }
        },
        {   "type": "function",
            "function": {
                "name": "ship_certification_status",
                "description" : "Tools provided by MCP to checks the certification status of a specified ship. Return the detail of expiry date of the certification, " \
                    "and the number of days left before the certification expires",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ship_name": {
                            "type": "string",
                            "description" : "The name of the ship for which the certification status is being checked",
                        },
                        "certification_type": {
                            "type": "string",
                            "description" : "The type of certification whose status is being checked. " \
                              "Should be one of ['Hull_Inspection', 'Safety_Equipment', 'Environmental_Compliance', 'Crew_Training']",
                        }
                    },
                    "required": ["ship_name", "certification_type"],
                },
            }
        },
        {   "type": "function",
            "function": {
                "name": "document_retriever_ship_certification",
                "description" : "Retrieves documents about detail of ship certifications based on a given query. " \
                    "It processes the retrieved documents, extracting the source, page, and content information from each document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description" : "A set of keyword/sentence representing the search query used to retrieve relevant documents",
                        },
                    },
                    "required": ["query"],
                },
            }
        },
        {   "type": "function",
            "function": {
                "name": "document_retriever_about_mcp",
                "description" : "Retrieves documents about detail of Maritime Cloud Platform (MCP) based on a given query. " \
                    "It processes the retrieved documents, extracting the source, and content information from each document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description" : "A set of keyword/sentence representing the search query used to retrieve relevant documents",
                        },
                    },
                    "required": ["query"],
                },
            }
        },
        {   "type": "function",
            "function": {
                "name": "document_retriever_about_bsk",
                "description" : "Retrieves documents about detail of Barata Sentosa Kencana Company (BSK), a company behind Maritime Cloud Platform (MCP) based on a given query. " \
                    "It processes the retrieved documents, extracting the source, and content information from each document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description" : "A set of keyword/sentence representing the search query used to retrieve relevant documents",
                        },
                    },
                    "required": ["query"],
                },
            }
        },
    ]
    return tools_list

def get_ship_travel_info(ship_name):
    """Get travel information of the desired ship."""

    dummy_origin_harbor_list = ["Pelabuhan Tanjung Mas, Semarang",
    "Pelabuhan Tanjung Intan, Cilacap,", "Pelabuhan Guluk, Sumenep",
    "Pelabuhan Tanjung Perak, Surabaya", "Pelabuhan Tanjung Priok, Jakarta"]

    dummy_destination_harbor_list = ["Pelabuhan Gilimanuk, Bali",
    "Pelabuhan Labuan Bajo, Lombok", "Pelabuhan Pulau Molas, Flores",
    "Pelabuhan Mamboro, NTT", "Pelabuhan Tobelo, Maluku Utara"]

    travel_info = {
        "ship_name" : ship_name,
        "harbor_origin": random.choice(dummy_origin_harbor_list),
        "harbor_destination": random.choice(dummy_destination_harbor_list),
        "estimated_time_of_arrival": str(datetime.datetime.now() + datetime.timedelta(hours = random.randint(26, 80))),
    }

    return json.dumps(travel_info)

def ship_certification_status(ship_name, certification_type):
    ship_certifications = { 
        'Hull_Inspection': {'expiry_date': datetime.date(2026, 6, 30)},
        'Safety_Equipment': {'expiry_date': datetime.date(2025, 12, 31)},
        'Environmental_Compliance': {'expiry_date': datetime.date(2023, 1, 15)},
        'Crew_Training': {'expiry_date': datetime.date(2024, 9, 30)},
    }
    
    # Fetch the expiry date for the given certification type
    today = datetime.date.today()
    certification_info = ship_certifications.get(certification_type, {})
    expiry_date = certification_info.get('expiry_date', 'Unknown')
    days_left = (expiry_date - today).days
    expiry_date_str = expiry_date.strftime('%Y-%m-%d')
    
    # Prepare the response dictionary
    certification_status = {
        "ship_name": ship_name,
        "certification_type": certification_type,
        "expiry_date": expiry_date.strftime('%Y-%m-%d') if isinstance(expiry_date, datetime.date) else expiry_date,
        "days_left_before_expired": f"{days_left} days"
    }
    
    # Convert the dictionary to JSON
    return json.dumps(certification_status)

def create_retriever(path, k_num):
    vector_db = Chroma(
        persist_directory = path,
        embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vector_db.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":k_num}
    )
    return retriever

def document_retriever_ship_certification(query):
    retriever = create_retriever("docs/Vectorstore_folder_dummycertification", 2)
    retrieved_docs = retriever.invoke(query)
    docs_list = []
    for retrieved_doc in retrieved_docs:
        retrieved_doc_metadata = retrieved_doc.metadata
        doc_json = {
            "source": retrieved_doc_metadata['source'],
            "page": retrieved_doc_metadata['page'],
            "content": retrieved_doc.page_content
        }
        docs_list.append(doc_json)
    return json.dumps(docs_list)

def document_retriever_about_mcp(query):
    retriever = create_retriever('docs/Vectorstore_folder_mcp', 2)
    retrieved_docs = retriever.invoke(query)
    docs_list = []
    for retrieved_doc in retrieved_docs:
        retrieved_doc_metadata = retrieved_doc.metadata
        doc_json = {
            "source": retrieved_doc_metadata['source'],
            "content": retrieved_doc.page_content
        }
        docs_list.append(doc_json)
    return json.dumps(docs_list)

def document_retriever_about_bsk(query):
    retriever = create_retriever("docs/Vectorstore_folder_BSK", 3)
    retrieved_docs = retriever.invoke(query)
    docs_list = []
    for retrieved_doc in retrieved_docs:
        retrieved_doc_metadata = retrieved_doc.metadata
        doc_json = {
            "source": retrieved_doc_metadata['source'],
            "content": retrieved_doc.page_content
        }
        docs_list.append(doc_json)
    return json.dumps(docs_list)