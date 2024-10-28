# Prompt_Engineer

Copyright (c) 2024, 
Inho Jake Park,
inhoparkloyal@gamil.com,
Gwangju Institute of Science and Technology,

All rights reserved.


This task is to analyze the traffic accident which is recored by a dashcam on the ego car.
The key purpose of this task is not to train the model but pesudo learning task. 

Several key tools have been used in this task such as: Prompt engineering, RAG, Vision-to-text(LLM), GPT API, etc.
Background of this research has begun from the consideration of lack amount of traffic accident video caption dataset.

Key contributions of this task are as follows:

1) Performance enhancing during inference.
2) Collecting high quality documents of traffic accident from national facilities.
3) Suggesting auto feedback prompting system, which can be highly accessible to any other models.

The code provided via Github:
1) chat_session.py --> process python file for auto feedback prompting system Which is you have to modify to fit into your own model.
2) gpt_rag.py --> After receiving the response from your model, this code will generate the "Hint" to your model
3) eval.py --> Exisiting method for this field is not accurate enough as no one can tell how GPT or any other LLM model works to evaluation; therefore, this evaluation method has been combined with n-gram and LLM method to migrate the risk of hallucintion caused by the LLM.
