# Prompt_Engineer

Copyright (c) 2024, 
Inho Jake Park,
inhoparkloyal@gamil.com,
Gwangju Institute of Science and Technology,

All rights reserved.

This project focuses on analyzing traffic accidents recorded by dashcams mounted on ego vehicles. The primary goal is not to train a model but to execute a pseudo-learning task.

Several key tools and techniques have been utilized, including prompt engineering, Retrieval-Augmented Generation (RAG), Vision-to-Text (LLM), and the GPT API. This research originated from the recognition of the limited availability of traffic accident video caption datasets.

Key contributions of this project include:

Enhancing performance during inference.
Collecting high-quality traffic accident documentation from national facilities.
Proposing an auto-feedback prompting system that can be applied flexibly to various models.
Code files in this repository:

chat_session.py – This script manages the auto-feedback prompting system. Modify it to align with your specific model.
gpt_rag.py – After your model generates a response, this script provides "hints" to guide your model’s improvements.
eval.py – Existing evaluation methods in this field lack accuracy due to the opaque nature of LLM behavior. To mitigate risks of hallucination, this evaluation approach combines n-gram analysis with LLM-based methods.
