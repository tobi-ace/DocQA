import streamlit as st
from io import BytesIO
from streamlit_pdf_viewer import pdf_viewer
import time
import torch


def main():
  st.set_page_config(layout="wide", page_title= "DocQA")
  st.title("Document Q/A with LLMs")
  st.caption("Chat with an LLM about a PDF document")

  with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF document", type= ('pdf'))

  col1, col2 = st.columns(2)

  container_height = 500

  with col1:
    with st.container(height=container_height):
      st.subheader("Document")
      if uploaded_file:
        pdf_viewer(uploaded_file.getvalue())

  with col2:
    messages = st.container(height=container_height)
    if "messages" not in st.session_state:
      st.session_state.messages = [{"role":"assistant", "content":"How can I help you"}]

    for message in st.session_state.messages:
      messages.chat_message(message["role"]).write(message["content"])

    if prompt := st.chat_input():
      st.session_state.messages.append({"role":"user", "content":prompt})
      messages.chat_message("user").write(prompt)

      if uploaded_file:
        #todo: validate question
        file_text = extract_text_from_pdf(uploaded_file.getvalue())
        
        QA_input = {
          'question': prompt,
          'context': file_text
        }

        model = load_model()

        content = model(QA_input)['answer']
        
        st.session_state.messages.append({"role":"assistant", "content":content})
        messages.chat_message("assistant").write(content)

      else:
        time.sleep(1)
        st.session_state.messages.append({"role":"assistant", "content":"Please upload a document!"})
        messages.chat_message("assistant").write("Please upload a document!")

def extract_text_from_pdf(ufile):
    from pypdf import PdfReader
    # pdf_file = ufile.get_value()
    pdf_reader = PdfReader(BytesIO(ufile))
    combined_text = ""

    for i,page in enumerate(pdf_reader.pages):  # Iterate through pages
      text = page.extract_text()  # Extract text from each page
      combined_text += f'{text}' 
    
    return combined_text 

def load_model():
  from transformers import pipeline

  device = 0 if torch.cuda.is_available() else -1
  model = 'distilbert/distilbert-base-cased-distilled-squad'

  return pipeline('question-answering', model=model, tokenizer=model, device=device)


if __name__ == "__main__":
  main()