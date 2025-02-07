# LLM Hugging Face Project
-----------------------------------------------------
### Project Requirements

    **Frontend:** React
    **Backend:** Node.js
    **ML Integration:** Hugging Face


### Installation & Setup

**Prerequisites**

    - Basics of Machine Learning
    - Hugging Face
    - Pandas, NumPy
    - React
    - Node.js, Express.js

**Steps**

    1. Clone the repository
    ```
    git clone https://github.com/techiescamp/mlops.git
    cd mlops/llm_huggingface
    ```

    2. Create a virtual environment (Optional but recommended)
    ```
    # for windows
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate     # For Windows
    ```

    3. Install dependencies (if requirements.txt exists)
    ```
    pip install -r requirements.txt
    ```

    4. After forking this repository and pull all the files.
        1. Install required packages each "backend" and "client" using below code 
        ``` npm install ```

        2. Start the application on each file using below command
        ``` npm start ```

### API Endpoints

Method         Endpoint                            Model                                                       Description
-----------------------------------------------------------------------------------------------------------------------------------
POST        /api/summarize                   facebook/bart-large-cnn                                      Summarizes the large text
 
POST        /api/text-generation             gpt2                                                         Generates text

POST        /api/update/text-generation      EleutherAI/gpt-neo-2.7B                                      Generates optimised text than above model

POST        /api/sentiment-analysis          Liusuthu/my_text_classification_model_based_on_distilbert    Classify `positive` and `negative` emotions

POST        /api/question-answer             distilbert/distilbert-base-cased-distilled-squad             Generates answer based on given context

POST        /api/table-question-answer       google/tapas-large-finetuned-wtq                             Generates answer based on given sql file - accepts oly `.csv`


### What is LLM ?
Large Language Models are advanced machine learning models designed to understand and generate human-like text. They are trained on massive amounts of text data, enabling them to perform tasks involve understanding and generation.  
    
LLMs use deep learning architectures, particularly transformer models, such as GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers) and others.
 
### Hugging Face ?
Hugging Face is an open-source platform and community that provides tools, libraries, and models for natural language processing (NLP) and machine learning. It is best known for its Transformers library, which allows developers to leverage state-of-the-art LLMs like BERT, GPT, and others.

## License

This project is open-source and available under the MIT License.
**&copy; www.techiescamp.com/**