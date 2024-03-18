import streamlit as st
#import subprocess
#import sys

import cv2
from ultralytics import YOLO

import numpy as np

import os

from groq import Groq

st.subheader('Vuoi sapere cosa vedo in questa foto?', divider='rainbow')

option = st.selectbox(
    '**Seleziona il modello di :blue[YOLOv8]** 👇',
    ('Object Detection', 'Instance Segmentation', 'Image Classification', 'Pose Estimation', 'Oriented Bounding Boxes Object Detection'))

st.write('**Hai selezionato il modello :rainbow[', option + ']**')

# Load the YOLOv8 model
if option == 'Object Detection':
    model = YOLO('yolov8n.pt') # Object Detection
elif option == 'Instance Segmentation':
    model = YOLO('yolov8n-seg.pt') # Instance Segmentation  
elif option == 'Image Classification':
    model = YOLO('yolov8n-cls.pt') # Image Classification     
elif option == 'Pose Estimation':
    model = YOLO('yolov8n-pose.pt') # Pose Estimation
elif option == 'Oriented Bounding Boxes Object Detection':
    model = YOLO('yolov8n-obb.pt') # Oriented Bounding Boxes Object Detection       
     
#model = YOLO('yolov8n.pt') # Object Detection
#model = YOLO('yolov8n-seg.pt') # Instance Segmentation
#model = YOLO('yolov8n-cls.pt') # Image Classification
#model = YOLO('yolov8n-pose.pt') # Pose Estimation
#model = YOLO('yolov8n-obb.pt') # Oriented Bounding Boxes Object Detection

img_file_buffer = st.camera_input("Adesso scatta una foto e ti dirò cosa vedo...")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    #success, frame = bytes_data.read()

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    #st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    #st.write(cv2_img.shape)
    #if success:
    # Run YOLOv8 inference on the frame
    results = model(cv2_img)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    #print('results: ', results)
    
    # Lettore QRCode
    text_color = (255, 255, 255)
    decoder = cv2.QRCodeDetector()
    data, points, _ = decoder.detectAndDecode(cv2_img)
    if points is not None:
        print('Decoded data: ' + data)
        #st.write('Decoded data: ' + data)
        qrcolor = (239, 108, 0)
        points = points[0]
        for i in range(len(points)):
            pt1 = [int(val) for val in points[i]]
            pt2 = [int(val) for val in points[(i + 1) % 4]]
            cv2.line(annotated_frame, pt1, pt2, (255, 152, 0), thickness=5)
        
        if data:
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            (w, h), _ = cv2.getTextSize("QRCode value: " + data, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # Prints the text    
            annotated_frame = cv2.rectangle(annotated_frame, (0, 0), (w, 30), qrcolor, -1)
            annotated_frame = cv2.putText(annotated_frame, "QRCode value: " + data, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            print("Trovato un QRCode contenente i seguenti dati: ", data)
            st.write("Trovato un QRCode contenente i seguenti dati: ", data)
            #st.write("[+] QR Code detected, data:", data)
    
    
    #st.write(results[0].plot())
    st.image(annotated_frame)

    # Display the annotated frame
    #cv2.imshow("YOLOv8 Inference", annotated_frame)

st.subheader('Prova a chattare con LLaMA2-70b', divider='rainbow')

# Groq API key
# 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

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Scrivi un breve messaggio motivazionale e di benvenuto di massimo 250 caratteri per un visitatore che entra in un ufficio di una software house",
        }
    ],
    model="llama2-70b-4096", #"mixtral-8x7b-32768",
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    #stream=True,
    stop=None,
)

print(chat_completion.choices[0].message.content)

# Insert a chat message container.
with st.chat_message("assistant"):
    st.write("Ciao 👋")
    st.write(chat_completion.choices[0].message.content)
    #st.line_chart(np.random.randn(30, 3))

# Display a chat input widget.
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")    





if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama2-70b-4096"

#message = st.chat_message("user")
#message.write("Hello")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Scrivi qualcosa..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["groq_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
        )
        response = st.write_stream(stream)
    print(response)    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
      