import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
name_model = model.name_or_path

# Page title
st.title('GPT-2 Model')

# Sidebar for input parameters
with st.sidebar:
    st.header('Model Parameters')
    temperature = st.slider('Temperature', min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_length = st.slider('Max Length', min_value=50, max_value=700, value=200, step=50)



# Main content
input_text = st.text_input('Type here')
logs = []

if st.button('Generate'):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    logs.append(f'Model Name: {name_model}')
    logs.append(f'Input Text: {input_text}')
    logs.append(f'Input IDs: {input_ids.tolist()}')
    logs.append(f'Temperature: {temperature}')
    logs.append(f'Max Length: {max_length}')

    start_time = time.time()
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, do_sample=True)
    end_time = time.time()

    execution_time = end_time - start_time
    logs.append(f'Execution Time: {execution_time} seconds')
    logs.append(f'Raw Output: {output.tolist()}')

    if output.shape[0] > 0:
        output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    else:
        output_text = ""

    st.write(f'{input_text}{output_text}')

    with st.sidebar:
        st.header('Logs')
        st.write('\n'.join(logs))