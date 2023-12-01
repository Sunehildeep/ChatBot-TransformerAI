# ChatBot using Transformer Encoder-Decoder Model

This project aims to create a chatbot using the Transformer encoder-decoder model, based on the groundbreaking "Attention Is All You Need" paper. The Transformer architecture has revolutionized natural language processing tasks, including machine translation and chatbot development. In this project, we leverage the power of self-attention mechanisms to build an intelligent and interactive chatbot.

## Features

- **Transformer Architecture:** The chatbot is built using the Transformer architecture, which allows it to capture contextual dependencies and generate accurate responses.
- **Self-Attention Mechanism:** The model utilizes self-attention mechanisms to attend to relevant parts of the input sequence, enabling it to understand the context and generate context-aware responses.
- **Multi-Head Attention:** Multiple attention heads are employed to capture different types of dependencies, resulting in a more comprehensive understanding of the input and improved response generation.
- **Encoder-Decoder Framework:** The chatbot follows the classic encoder-decoder framework, where the encoder processes the input sequence and the decoder generates the response based on the encoded representation.


## Requirements

- Python 
- Tensorflow library (version 2.10.0)

## Installation

1. Clone the repository
2. Install the required packages
3. I am using tensorflow-gpu 2.10.0 for this project
4. To train the model, edit the preprocess.py file and replace your dataset with your own. 
5. Run the main.py file to start the training process. You can play around with the hyperparameters to best fit your needs.
6. Run the chat.py file to load the trained model :)
   
![image](https://github.com/Sunehildeep/ChatBot-TransformerAI/assets/23412507/a7120373-9fad-4ab2-8ee9-2d67347e064b)
