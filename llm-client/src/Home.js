import React from 'react'

const Home = () => {
  return (
    <div>
        <h1>Welcome to <span>Large Language Model</span> Projects </h1>
        <div className='home-content'>
            <h3>What is LLM ?</h3>
            <p>
                <b>Large Language Models</b> are advanced machine learning models designed to understand and generate 
                human-like text. They are trained on massive amounts of text data, enabling them to perform tasks involve
                understanding and generation.  
            </p>
            <p>
                LLMs use deep learning architectures, particularly transformer models, such as GPT (Generative Pre-trained Transformer),
                BERT (Bidirectional Encoder Representations from Transformers) and otehrs.
            </p>

            <h3>Hugging Face ?</h3>
            <p>
            Hugging Face is an open-source platform and community that provides tools, libraries, and models for natural language processing (NLP) and 
            machine learning. It is best known for its Transformers library, which allows developers to leverage state-of-the-art LLMs like BERT, GPT, and others.
            </p>
        </div>
    </div>
  )
}

export default Home