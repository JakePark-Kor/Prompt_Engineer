"""
 Copyright (c) 2024, 
 Gwangju Institute of Science and Technology,
 OpenGV Lab
 All rights reserved.
"""
import torch
from utils.config import Config
from models.videochat2_it import VideoChat2_it
from conversation import Chat
from utils.easydict import EasyDict
import json
import os
from copy import deepcopy

import torch.nn.functional as F

from peft import get_peft_model, LoraConfig, TaskType

# RAG

from gpt_rag import analyze_with_rag, generate_image_summary

# Cosine Similarity

from transformers import AutoTokenizer, AutoModel
#from sklearn.metrics.pairwise import cosine_similarity

# Entropy

#from scipy.stats import entropy

import matplotlib.pyplot as plt

# Set environment variables to suppress TensorFlow warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set PyTorch memory management environment variables

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# ========================================
#           Model Initialization
# ========================================

def init_model():
    
    print('Initializing VideoChat2_it')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    model = VideoChat2_it(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.float()  
    model = model.eval()
    
    # add lora to run stage3 model
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.
    )
    
    model.llama_model = get_peft_model(model.llama_model, peft_config)

    state_dict = torch.load("/home/workspace/data/videochat2_7b_stage3.pth", "cpu")
    
    if 'model' in state_dict.keys():
        
        msg = model.load_state_dict(state_dict['model'], strict=False)
        
    else:
        
        msg = model.load_state_dict(state_dict, strict=False)
        
    print(msg)

    model = model.eval()
    chat = Chat(model)
    
    print('Initialization Finished')
    
    return chat

# ========================================
#            Core Functionality
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAG PDF data preparation and feedback loop
pdf_path = "/home/workspace/data/NHTHA/"
pdf_files = [os.path.join(pdf_path, file) for file in os.listdir(pdf_path) if file.endswith('.pdf')]

tokenizer = None
model = None

def load_model_and_tokenizer():
    
    global tokenizer, model
    
    if tokenizer is None:
    
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    
    if model is None:
    
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)

response_similarities = []
feedback_similarities = []
combined_similarities = []

def get_embedding(text):
    
    load_model_and_tokenizer()
    
    if isinstance(text, list):
        
        text = " ".join(text)
        
    elif not isinstance(text, str):
        
        raise ValueError("Input must be a string or a list of strings.")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        
     outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = embeddings.squeeze()  
    
    torch.cuda.empty_cache() 
    
    return embeddings

def calculate_cosine_similarity(text1, text2):
    
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    if len(embedding1.shape) == 3:
        
            embedding1 = embedding1.view(-1, embedding1.size(-1))  # (batch_size * sequence_length, hidden_size)
        
    if len(embedding2.shape) == 3:
        
            embedding2 = embedding2.view(-1, embedding2.size(-1))
    
    cosine_sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=-1)

    del embedding1, embedding2 
    
    torch.cuda.empty_cache() 
    
    
    return 1 - cosine_sim.item()

# ========================================
#       Feedback Application Logic
# ========================================

def should_add_feedback(feedback_similarity, threshold, patience_counter, patience_limit):
    
    if feedback_similarity > threshold:
        
        patience_counter = 0  
        
        return True, patience_counter
    
    else:
        
        patience_counter += 1
        
        if patience_counter >= patience_limit:
            
            print(f"Model has converged after {patience_counter} iterations.")
            
            return False, patience_counter  
        
        return False, patience_counter

def get_entropy(logits):
    
    """
    주어진 텍스트에 대해 언어 모델을 사용해 확률 분포를 계산하고, 이를 바탕으로 엔트로피를 반환하는 함수
    """

    logits = logits[:, :-1]  # 마지막 토큰 제외 (일반적으로 [PAD] 토큰일 가능성이 있음)
    
    entropy = calculate_entropy_from_logits(logits) 
    
    return entropy

def calculate_entropy_from_logits(logits):
    
    """
    주어진 logits에서 확률을 계산하고 이를 바탕으로 엔트로피를 계산함.
    """
    
    probs = F.softmax(logits, dim=-1)
    
    # 엔트로피 계산: -sum(p * log(p))
    
    entropy_values = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # 각 토큰의 엔트로피 평균을 반환
    
    return entropy_values.mean().item()

def combined_similarity(previous_feedback, improved_feedback):
    
    feedback_similarity = calculate_cosine_similarity(previous_feedback, improved_feedback)
    
    print(f"Feedback Cosine Similarity: {feedback_similarity}")
    
    #response_entropy_prev = get_entropy(previous_logits)
    #response_entropy_improved = get_entropy(improved_logits)
    #response_entropy_change = abs(response_entropy_prev - response_entropy_improved)
    #print(f"response entropy : {response_entropy_change}")
    #combined_similarity_score =  feedback_similarity
    #print(f"Combined Similarity with Entropy: {combined_similarity_score}")
    #response_similarities.append(response_similarity)
    
    feedback_similarities.append(feedback_similarity)
    
    #combined_similarities.append(combined_similarity)
    
    return feedback_similarities

def plot_similarity_graph(video_name,path):
    
    """
    저장된 유사도 값을 기반으로 반복 횟수에 따른 코사인 유사도 변화를 그래프로 시각화하는 함수
    """

    iterations = range(1, len(response_similarities) + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, response_similarities, label="Response Cosine Similarity", marker='o')
    plt.plot(iterations, feedback_similarities, label="Feedback Cosine Similarity", marker='s')
    plt.plot(iterations, combined_similarities, label="Combined Cosine Similarity", marker='x')
    
    plt.title("Cosine Similarity Across Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)  # 0 ~ 1
    plt.legend()
    plt.grid(True)
    
    graph_path = f"{path}/{video_name}_cos.png"
    
    
    plt.savefig(graph_path)
    
    print(f"Graph saved as {graph_path}")
    
    plt.close()

# ========================================
#        Question State Management
# ========================================

class QuestionState:
    
    def __init__(self, question):
        
        self.question = question
        self.initial_response = None   # Store initial response
        self.improved_responses = []   # Store all improved responses
        self.feedbacks = []            # Store all feedbacks
        self.similarities = []         # Track similarity scores
        self.patience = 0              # Track patience counter
        self.improved_questions = []    # Store the improved question after feedback

    def update_response(self, response, is_initial=False):
        
        if is_initial:
            
            self.initial_response = response
            
        else:
            
            self.improved_responses.append(response)
        
    def add_feedback(self, feedback):
        
        self.feedbacks.append(feedback)
        
    def update_improved_question(self, improved_question):
        
        self.improved_questions.append(improved_question)  # Store all improved questions

def process_video_files(video_directory, config, chat):
    
    video_files = [f for f in os.listdir(video_directory) if f.endswith(('.mp4', '.avi', 'webm'))] 
    
    for index, video_file in enumerate(video_files):
        
        video_path = os.path.join(video_directory, video_file)
        
        print(f"Processing video: {video_path}")
        
        video_name = os.path.splitext(video_file)[0]         
        
        ans_generation(video_path, config, chat, video_name)

        torch.cuda.empty_cache() 
        
def ans_generation(video_path, config, chat, video_name):
    
    # Initialize patience_counter and config parameters
    
    threshold = config['threshold']
    patience_limit = config['patience_limit']
    
    conv_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    
    img_list = []
    
    message, img_list, conv_state, frames = chat.upload_video(video_path, conv_state, img_list, 4)
    
    img_summary_text = None
    
    if config["use_image_summaries"] == True:
              
        img_summary_text = generate_image_summary(frames)
        
    conv_state_feedback = conv_state.deepcopy()
    
    print("Video Upload Response:", message)

    questions = [
        "As a AI traffic accient analyst, please diagnose the video and extract the external conditions based on the following:\n - Weather: What are the weather conditions (e.g., clear, rainy, foggy)?\n - Road Conditions: Is the road wet, dry, or icy?\n- Road Type: Is the road a highway, rural road, or intersection?\n- Number of Lanes: How many lanes are visible on the road?\n- Lighting: Describe the lighting conditions (e.g., daylight, night, streetlights). Your answer should follow the format which is given in the question.",
        "Identify all visible vehicles in the video:\n- Type: What type of vehicle is visible  (e.g., sedan, SUV, truck)? And Count them. \n- Position: Where is the vehicle positioned on the road (e.g., front, middle, rear)?",
        "Analyze the vehicle movements based on the video:\n- Braking: Did the vehicle apply brakes? (Yes or No). If Yes, describe when and how strongly.\n- Acceleration: Did the vehicle accelerate before the crash? (Yes or No). If Yes, describe the timing and speed increase.\n- Steering: Did the vehicle make any steering adjustments? (Yes or No). If Yes, describe the direction and angle.\n- Lane Change: Did the vehicle change lanes? (Yes or No). If Yes, describe the timing and number of lanes changed.\n- Evasive Maneuvers: Did the vehicle attempt to avoid the crash (e.g., swerving, sudden braking)? Describe the actions taken.\n- Vehicle Stability: Was the vehicle stable before the crash (e.g., no skidding or drifting)? (Yes or No). \n If No, describe the instability. Your answer should follow the format which is given in the question",
        "Analyze the collision points, if any:\n- Front: Was there an impact to the front of the vehicle? (Yes or No)\n- Rear: Was there an impact to the rear of the vehicle? (Yes or No)\n- Side: Was there an impact to the side of the vehicle? (Yes or No)\n Your answer should follow the format which is given in the question.",
        "Analyze the events leading up to, during, and after the crash:\n- Pre-Crash: Describe what happened before the crash, including speed, vehicle behavior, and environmental factors.\n- Crash: If a crash occurred, describe the impact and the speed/direction of the vehicles. If no crash occurred, indicate this.\n- Post-Crash: Explain the vehicle's movement after the crash and whether it collided with any roadside objects (e.g., barriers, trees). For each state, your response should be around 150 tokens."
    ]
    question_config =[
        {"max_new_tokens": 150, 'num_beams': 1, 'min_length': 1},
        {"max_new_tokens": 150, 'num_beams': 1, 'min_length': 1},
        {"max_new_tokens": 150, 'num_beams': 1, 'min_length': 1},
        {"max_new_tokens": 150, 'num_beams': 1, 'min_length': 1},
        {} # Default value written in config
    ]
    
    
    question_states = [QuestionState(question) for question in questions]

    for iteration in range(1, config['feedback_iterations'] + 1):
        
        print(f"iteration : {iteration}")

        for i, q in enumerate(question_states):
            
            current_question_config = question_config[i]
            max_new_tokens = current_question_config.get('max_new_tokens', config['max_new_tokens'])
            num_beams = current_question_config.get('num_beams', config['num_beams'])
            min_length = current_question_config.get('min_length', config['min_length'])
            
            if iteration == 1:
                
                conv_state = chat.ask(q.question, conv_state)
                
                initial_response, _, conv_state, initial_logits = chat.answer(conv_state,
                                                        img_list,
                                                        max_new_tokens = max_new_tokens,
                                                        num_beams=num_beams,
                                                        min_length= min_length,
                                                        top_p=config['top_p'],
                                                        repetition_penalty=config['repetition_penalty'],
                                                        length_penalty=config['length_penalty'],
                                                        temperature=config['temperature'])
            

                # Store initial question and response
        
                q.update_response(initial_response, is_initial=True)

                print(f"Initial Response for question {i+1}: {initial_response}")
                
                feedback = analyze_with_rag(pdf_files=pdf_files, responses=[q.initial_response], 
                                                              img_list=frames, 
                                                              image_summaries=img_summary_text,
                                                              question_types= question_states)
                
                q.add_feedback(feedback)
                
                improved_question = f"{q.question} : Additionally, consider the following feedback: {feedback}"
                
                # Store initial improved_question (same as the original question for the first iteration)
                
                q.update_improved_question(improved_question)     
                
                torch.cuda.empty_cache()          
            
            else:
                
                feedback = analyze_with_rag(pdf_files=pdf_files, 
                                            responses=[q.improved_responses[-1] if q.improved_responses else q.initial_response], 
                                            img_list=frames, image_summaries= img_summary_text, question_types= question_states)
                
                combined_sim = combined_similarity(q.feedbacks[-1] if q.feedbacks else "", feedback)
                
                add_feedback, q.patience = should_add_feedback(combined_sim[-1], threshold, q.patience, patience_limit)

                if add_feedback:
                    
                    q.add_feedback(feedback)
                    
                    improved_question = f"{q.question} : Additionally, consider the following feedback: {q.feedbacks}"
                    
                    q.update_improved_question(improved_question)
        
                    conv_state = chat.ask(q.improved_questions[-1], conv_state_feedback)
                    
                    improved_response, _, conv_state, improved_logits = chat.answer(
                                                               conv_state,
                                                               img_list,
                                                               max_new_tokens=max_new_tokens,
                                                               num_beams=num_beams,
                                                               min_length=min_length,
                                                               top_p=config['top_p'],
                                                               repetition_penalty=config['repetition_penalty'],
                                                               length_penalty=config['length_penalty'],
                                                               temperature=config['temperature'])


                    q.update_response(improved_response)
                    
                    print(f"Improved Response for question {i+1}: {improved_response}")
                    
                    torch.cuda.empty_cache()
                
                else:
                    
                    print(f"Feedback not applied for question {i+1}. Patience: {q.patience}")
                    
                    q.update_improved_question(q.improved_questions[-1] if q.improved_questions else q.question)   
                    
                    torch.cuda.empty_cache()
                      
            if q.patience >= patience_limit:
                
                print(f"Question {i+1} has converged. Stopping feedback loop.")
                torch.cuda.empty_cache()
                break
            
    json_data = {"Video_path": config["data_path"], 
                 "Video_Name":video_name, 
                 "Iterations": [], 
                 "Image_Summary" : img_summary_text
                }
    
    for i, q in enumerate(question_states):
        
        iteration_data = {
            "Question": q.question,
            "Initial_Response": q.initial_response,
            "Improved_Questions": q.improved_questions,  # Store all improved questions
            "Feedbacks": q.feedbacks,
            "Improved_Responses": q.improved_responses
        }
        json_data["Iterations"].append(iteration_data)
        
    json_file_path = os.path.join(config["chat_saving_path"], f'feedback_session_{video_name}.json')   
    
    with open(json_file_path, 'w') as json_file:
        
        json.dump(json_data, json_file, indent = 4)
        
    print(f"Chat session saved to {json_file_path}")

    del img_list, frames  
    
    torch.cuda.empty_cache()




def main(config):
    
    chat = init_model()
    video_directory = config['data_path']
    process_video_files(video_directory, config, chat)

if __name__ == "__main__":
    
    config = {
        'max_new_tokens':500,
        'num_beams': 3,
        'min_length': 1,
        'top_p': 0.75,
        'repetition_penalty': 1.0,
        'length_penalty': 1.0,
        'temperature': 0.7,
        'data_path': "/home/workspace/Ask-Anything/sample/",
        'chat_saving_path': '/home/workspace/Ask-Anything/video_chat2/test/',
        'use_image_summaries': True,
        'feedback_iterations': 5,
        'threshold': 0.05,
        'patience_limit': 3
    }
    
    main(config)
    

    
