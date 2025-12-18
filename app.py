import streamlit as st
import cv2
from ultralytics import YOLO
import time

# ---------------------------------------------
# Streamlit setup
# ---------------------------------------------
st.set_page_config(page_title="E-Waste Object Detection", layout="centered")
st.title("E-Waste Object Detection using YOLOv8")

st.info(
    "This system detects and localises e-waste items in real time "
    "using a YOLOv8 object detection model."
)

# ---------------------------------------------
# Session state init
# ---------------------------------------------
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# ---------------------------------------------
# Load YOLO model
# ---------------------------------------------
model = YOLO("best.pt")

# ---------------------------------------------
# UI control (ONLY ONCE)
# ---------------------------------------------
st.session_state.run_detection = st.checkbox(
    "Start Live Detection",
    value=st.session_state.run_detection
)

frame_placeholder = st.empty()

# ---------------------------------------------
# Webcam + detection loop
# ---------------------------------------------
if st.session_state.run_detection:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not accessible.")
    else:
        st.success("Live detection running. Uncheck to stop.")

        while st.session_state.run_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame.")
                break

            results = model(frame, verbose=False)
            annotated_frame = frame.copy()

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                # Confidence threshold
                if conf < 0.5:
                    label = "Unknown E-Waste"

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            frame_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB"
            )

            time.sleep(0.03)

        cap.release()
        st.warning("Detection stopped.")
