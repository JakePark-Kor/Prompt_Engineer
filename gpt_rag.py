"""
 Copyright (c) 2024, 
 
 Gwangju Institute of Science and Technology,
 This code is based on "https://github.com/teddylee777/langchain-kr"
 
 All rights reserved.
"""

import time
import gc
import os
import json

from langchain.chains import RetrievalQA 

#from langchain_community.document_loaders import PyMuPDFLoader

from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain_chroma import Chroma

from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import hashlib
import openai
from transformers import AutoTokenizer, AutoModel

import torch
import numpy as np
import base64
import io
from PIL import Image
import requests

# OpenAI API 


# For the sake of keeping your privacy, type your own api key on the terminal with the code ==> [export OPENAI_API_KEY='your-openai-api-key']

os.environ['OPENAI_API_KEY'] = ''

openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    
    raise ValueError("OPENAI_API_KEY has not been set.")

pdf_path = "/home/workspace/data/NHTHA/"
pdf_files = [os.path.join(pdf_path, file) for file in os.listdir(pdf_path) if file.endswith('.pdf')]
cache_path = "/home/workspace/Ask-Anything/pdf_cache2/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None
vectorstore = None


def load_model():
    
    global model
    global tokenizer
    
    if model is None:
        
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    
    if tokenizer is None:
        
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
    return model


def optimize_memory():
    
    torch.cuda.empty_cache() 
    
    gc.collect()


# PDF Cache

def split_texts(texts, chunk_size=1000, chunk_overlap=200):
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    
    for text in texts:
        
        chunks = splitter.split_text(text)
        split_docs.extend(chunks)
        
    return split_docs


def get_pdf_cache_path(pdf_file):
    
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    
    return os.path.join(cache_path, f"{base_name}_cache.json")


def save_cached_data(pdf_file, data):
    
    cache_path = get_pdf_cache_path(pdf_file)
    
    with open(cache_path, 'w') as f:
        
        json.dump(data, f)

# PDF Partition_and_load

def load_cached_data(pdf_file):
    
    cache_path = get_pdf_cache_path(pdf_file)
    
    if os.path.exists(cache_path):
        
        try:
            
            with open(cache_path, 'r') as f:
                
                return json.load(f)
            
        except json.decoder.JSONDecodeError:

            print(f"Warning: Cache file {cache_path} is invalid. Re-extracting PDF: {pdf_file}.")
            
            return None 
        
    return None

def partition_and_load_pdfs(pdf_files):
    
    all_texts = []
    
    for pdf_file in pdf_files:
        
        cached_data = load_cached_data(pdf_file)
        
        if cached_data:

            all_texts.extend(cached_data)
            
        else:
            
            elements = partition_pdf(
                filename=pdf_file,
                extract_images_in_pdf=False,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=2000,
                new_after_n_chars=1800,
                combine_text_under_n_chars=1000
            )
            print(f"Extracted {len(elements)} elements from PDF: {pdf_file}.")

            texts = [el.text for el in elements if hasattr(el, 'text') and el.text]

            save_cached_data(pdf_file, texts)

            all_texts.extend(texts)
    
    split_text = split_texts(all_texts)
    
    optimize_memory()
    
    return split_text


def embed_text(text):
    
    load_model()
    
    with torch.no_grad():
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  
        model.to('cpu')
        
        torch.cuda.empty_cache()
        
    return embeddings

def embed_documents(documents, batch_size=16):
    
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        
        batch_docs = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)
            
        model.to('cpu')
        optimize_memory() 
        
    return torch.cat(all_embeddings, dim=0)

# Text to Documents

def convert_texts_to_documents(texts):
    
    documents = [Document(page_content=text) for text in texts]
    
    return documents

# Vectorstore initialization

def initialize_vectorstore(embeddings, persist_directory=None):
    
    global vectorstore
        
    if vectorstore is None:
        
        vectorstore = Chroma(collection_name="rag_multiple_pdfs",
                             embedding_function=embeddings, 
                             persist_directory=persist_directory)
      
        print("Vectorstore initialized")
        
    else:
        
        print("Vectorstore already initialized")
   
    return vectorstore

# Add texts from pdf to the vector store

#[OPENAI Embeddings]

def hash_text(text):
    
    """문서 내용을 기반으로 해시를 생성하여 고유 ID로 사용"""
    
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def add_texts_to_vectorstore(vectordb, texts, batch_size=1000, embeddings=None, persist_directory="/tmp/chromadb"):
    
    if embeddings is None:
        
        raise ValueError("Embeddings must be provided")

    documents = convert_texts_to_documents(texts)

    existing_doc_ids = set(vectordb._collection.get()['ids']) if vectordb is not None else set()

    for i in range(0, len(documents), batch_size):
        
        batch_documents = documents[i:i + batch_size]

        new_documents = []
        new_ids = []

        for doc in batch_documents:
            
            document_id = hash_text(doc.page_content) 

            if document_id not in existing_doc_ids:
               
                new_documents.append(doc)
                new_ids.append(document_id)
                existing_doc_ids.add(document_id)

        if not new_documents:
            
            print(f"Batch {i // batch_size + 1} already exists, skipping.")
            
            continue

        retries = 0
        max_retries = 10

        while retries < max_retries:
            
            try:
                
                vectordb.add_documents(new_documents, ids=new_ids)
                print(f"Batch {i // batch_size + 1} added successfully.")
                break
            
            except Exception as e:
                
                retries += 1
                print(f"Error occurred: {str(e)}. Retrying {retries}/{max_retries}...")
                time.sleep(70)
                gc.collect()

        if retries == max_retries:
            
            print(f"Failed to add batch {i // batch_size + 1} after {max_retries} retries.")
            break

    vectordb.persist()
 

# LLM initialization (GPT)

def initialize_llm(model_name="gpt-4o-mini"):
    
    return ChatOpenAI(temperature=0, max_tokens=300, model=model_name)


def tensor_to_base64_image(tensor):
    
    if isinstance(tensor, torch.Tensor):
        
        image_np = tensor.detach().cpu().numpy()
        
    elif isinstance(tensor, np.ndarray):
        
        image_np = tensor
        
    elif isinstance(tensor, Image.Image):  
        
        image_np = np.array(tensor)
        
    else:
        
        raise TypeError("Expected input to be a PyTorch Tensor, numpy array, or PIL Image")

    if len(image_np.shape) == 2:
        
        image_np = np.stack([image_np] * 3, axis=-1)  # [H, W] -> [H, W, 3]
    
    elif len(image_np.shape) == 3 and image_np.shape[-1] != 3:
        
        raise ValueError(f"Invalid image shape: {image_np.shape}. Expected (H, W, 3).")
    
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)

    buffered = io.BytesIO()
    
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return base64_image

def generate_image_summary(img_list):
    
    base64_images = []

    for i, tensor in enumerate(img_list):
        
        base64_image = tensor_to_base64_image(tensor)
        base64_images.append(f"data:image/png;base64,{base64_image}")
    
    images_payload = [{"type": "image_url", "image_url": {"url": img}} for img in base64_images]
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are looking at the view from an ego car. These frames depict a traffic accident. Provide a detailed summary and analysis of the video."},
            {
                "role": "user", 
                "content": images_payload  
            }
        ],
        "max_tokens": 500
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", 
        headers={"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"},
        json=payload
    )

    response.raise_for_status() 

    summary = response.json()["choices"][0]["message"]["content"].strip()

    print(f"Summarized Final Summary: {summary}")

    return summary

def generate_feedback_prompt(question_type, response):
    
    if question_type == "Pre-Crash":
        
        return f"""Please provide feedback on the following response: {response}
                    Ensure that the response includes details on road conditions, weather, lighting, and vehicle behavior before the crash.
                    Limit feedback to tokens."""

    elif question_type == "Lane Positioning":
        
        return f"""Please provide feedback on the following response: {response}
                    Ensure that the response describes the lane positions, road layout, and whether vehicles stayed in or crossed lanes before the crash.
                    Limit feedback to 200 tokens."""

    elif question_type == "Crash Phase":
        
        return f"""Please provide feedback on the following response: {response}
                    Ensure the description of speed changes, steering movements, and the exact point of impact.
                    Limit feedback to 200 tokens."""

    elif question_type == "Type of Collision":
        
        return f"""Please provide feedback on the following response: {response}
                    Ensure that the collision type is clearly identified (e.g., head-on, rear-end) based on vehicle positions after the crash.
                    Limit feedback to 200 tokens."""

    else:
        
        return f"""Please provide feedback on the following response: {response}

                                    The visual context provided above is the same video that another model saw and provided the response. 
                                    However, we need to ensure the model's response is accurate and complete. 
                                    
                                    Check if the following aspects are properly addressed:

                                    Pre-Crash:

                                    Environment and Vehicles: Ensure details on road conditions (e.g., wet, dry), weather, lighting, and surrounding features (e.g., barriers, trees) are included​(813528)​(813524).
                                    Vehicle Behavior: Assess whether the vehicle speeds, lane positions, and any evasive actions are clearly described.
                                    Lane Positioning:

                                    Road Layout: Is the exact number and type of lanes (e.g., divided highway, turn lanes) clear? Does the description capture whether the vehicles stayed in or crossed lanes before the crash?.
                                    Crash Phase:

                                    Collision Dynamics: Ensure the model describes speed changes, steering movements, and the exact point of impact (e.g., front, rear).
                                    Type of Collision:

                                    Ensure the model identifies the collision type (e.g., head-on, rear-end)  . If lacking, guide the model to base its identification on visual signs like vehicle damage or positions after the crash.
                                                                        
                                    **Give the model constructive feedback** that helps it refine its answer. Provide **hints or suggestions** to improve the accuracy and completeness of the response. 
                                    Additionally, keep in mind that the model only has access to visual data, and does not have information about audio or data from the Event Data Recorder (EDR). 
                                    Limit your feedback to 200 characters.
                                    """


def analyze_with_rag(pdf_files, responses, img_list = None, image_summaries = None, question_types = None):
    
    feedbacks = []
    
    img_summary_text = ""
    
    if image_summaries is not None:
        
        img_summary_text = " ".join(image_summaries)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    #embeddings = embed_text
    
    vectorstore = initialize_vectorstore(embeddings=embeddings,persist_directory="chroma_db")

    texts = partition_and_load_pdfs(pdf_files)
    
    add_texts_to_vectorstore(vectorstore, texts, embeddings=embeddings)
    
    torch.cuda.empty_cache()

    llm = initialize_llm()

    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                           chain_type="map_reduce", 
                                           retriever=vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 10})
                                           )

    for i, response in enumerate(responses):
        
        question_type = question_types[i] if question_types else "General"
            
        feedback_question = generate_feedback_prompt(question_type, response)
                                    
        if img_summary_text:
                
            full_question = f"Visual context from the images: {img_summary_text}\n\n{feedback_question}"
                
        else:
                
            full_question = feedback_question
        

        feedback = qa_chain.run(full_question)
            
        feedbacks.append(feedback)
        
        optimize_memory()
    
    return feedbacks
