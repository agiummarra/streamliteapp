# 1. Setup Iniziale
import streamlit as st
import cv2
#from yolov5.detect import YOLOv5Detector  # Sostituisci con la tua implementazione per YOLOv8/v9
#import psycopg2
import requests
from ultralytics import YOLO
from camera_input_live import camera_input_live

st.set_page_config(
    page_title="Software House Presence Detection",
    page_icon="🤖",
)

# 2. Interfaccia Grafica con Streamlit
st.title("Software House Presence Detection")
st.header("Welcome!")

st.sidebar.markdown("# 🤖 :rainbow[YOLO] ✨")
st.sidebar.markdown("# 🧠 :blue[Model parameters] 🪄")

option = st.sidebar.selectbox(
    '**Seleziona il modello di :blue[YOLOv8]** 👇',
    ('Object Detection', 'Instance Segmentation', 'Image Classification', 'Pose Estimation', 'Oriented Bounding Boxes Object Detection'))

st.sidebar.write('**Hai selezionato il modello :rainbow[', option + ']**')

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
     

# 3. Riconoscimento di Oggetti con YOLO
#detector = YOLOv5Detector()  # Inizializza il rilevatore YOLO
cap = cv2.VideoCapture(0)
#image = camera_input_live()

#if image:
#    st.image(image)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #st.markdown(results[0])
        
        #st.text_area("Output", results[0])
        # Display the annotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)
        #st.camera_input("YOLOv8 Inference", annotated_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# 4. Chiamata API a Groqcloud
# Implementa la chiamata API a Groqcloud per ottenere il messaggio motivazionale personalizzato

# 5. Gestione del QRCode
# Implementa la lettura del QRCode dalla fotocamera e il confronto con il codice memorizzato

# 6. Gestione delle Presenze e Motivazioni
# Implementa la registrazione dell'ingresso/uscita e la gestione delle motivazioni

# 7. Risposta Audio e Visiva
# Visualizza il messaggio di benvenuto e motivazionale
# Riproduci il messaggio audio attraverso gli speaker del dispositivo
