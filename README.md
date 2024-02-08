# Question-Answering on PDF

## Description
A small fun Streamlit app I made to practice usage of LLM, particularly T5 base model [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M). Providing QA on any PDF file.

## Installation
Clone this repository and install the requirements
```bash
pip install -r requirements.txt
```
Clone the model repository
```bash
git lfs install
git clone https://huggingface.co/MBZUAI/LaMini-T5-738M
```

## Usage
Place your PDF inside ```docs``` folder

Run ```ingest.py``` to create embeddings and store vector database
```bash
python ingest.py
```
Afterward, run ```app.py``` to set up Streamlit app for inference
```bash
streamlit run app.py
```
