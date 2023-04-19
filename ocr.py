import easyocr as ocr
import streamlit as st
from PIL import Image
import numpy as np
import torch
import docx
import base64


# Load the model and specify to use CPU if CUDA is not available
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = torch.load("tamil_net.pt")
else:
    device = torch.device('cpu')
    model = torch.load("tamil_net.pt", map_location=device)

# Load the reader
reader = ocr.Reader(["en", "ta"], model_storage_directory=".", gpu=False)

# Title
st.title("Extract Text from Images")

# Subtitle
st.markdown("## using EASYOCR")

st.markdown("")

# Image uploader
image = st.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'])

@st.cache_data
def load_model():
    reader = ocr.Reader(['en', 'ta'], model_storage_directory='.')
    return reader

reader = load_model() #load model

from PIL import Image
im = Image.open("tamilimg.jpeg")

im

reader = ocr.Reader(['en','ta'],model_storage_directory='.')

bounds = reader.readtext("tamilimg.jpeg")

from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image,bounds, color = 'blue', width =2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0,p1,p2,p3 = bound[0]
        draw.line([*p0,*p1,*p2,*p3,*p0], fill=color, width = width)
    return image

draw_boxes(im, bounds)

if image is not None:
    input_image = Image.open(image) # Read the uploaded image
    st.image(input_image) # Display the uploaded image

    with st.spinner("🤖 AI is at Work! "):
        result = reader.readtext(np.array(input_image))
        result_text = [] # empty list for results

        for text in result:
            result_text.append(text[1])

        st.write(result_text)

        # Create a new word document
        doc = docx.Document()

        # Write the extracted text to the document
        for text in result_text:
            doc.add_paragraph(text)

        # Save the document
        doc_filename = 'extracted_text.docx'
        doc.save(doc_filename)

        # Display a link to the saved document
        st.markdown(f"Download the extracted text: <a href='data:application/octet-stream;base64,{base64.b64encode(open(doc_filename, 'rb').read()).decode('utf-8')}' download='{doc_filename}'>{doc_filename}</a>", unsafe_allow_html=True)

    st.balloons()
else:
    st.write("Upload an Image")

st.caption("SIHAAM")
