Traffic Accident Analysis via Dashcam Footage With Effective Prompt

Copyright (c) 2024, 
Inho Jake Park,
inhoparkloyal@gamil.com,
Gwangju Institute of Science and Technology,

**All rights reserved.**

This project aims to analyze traffic accidents recorded by dashcams on ego vehicles. The primary goal is not to train a model, but rather to execute a pseudo-learning task that focuses on inference and analysis.

## Overview

This task incorporates advanced AI techniques and tools, including prompt engineering, Retrieval-Augmented Generation (RAG), Vision-to-Text using LLMs, and the GPT API. The research was motivated by the limited availability of comprehensive traffic accident video caption datasets.

## Key Contributions

1. **Performance Enhancement during Inference**: Optimized processes to improve model performance when processing traffic accident data.
2. **High-Quality Documentation Collection**: Curated high-quality traffic accident documentation from national facilities.
3. **Auto-Feedback Prompting System**: Developed a flexible, automatic feedback system that can be easily integrated with other models.

## Code Files

- **`chat_session.py`**: Main script for managing the auto-feedback prompting system. Modify this file to align with your specific model requirements.
- **`gpt_rag.py`**: Generates "hints" or feedback based on your model's response to guide improvements in output.
- **`eval.py`**: Combines n-gram and LLM-based evaluation methods to enhance accuracy. This approach mitigates the risk of hallucinations common with LLMs, as traditional evaluation methods are often insufficient for this purpose.

---

Feel free to reach out if you have any questions or need further customization for your model setup.

