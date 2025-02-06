import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime
from PIL import Image
import cv2

# Load Model & Feature Columns
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit Page Config
st.set_page_config(page_title="Chemical Dosage Prediction", page_icon="🧪", layout="wide")

# Title
st.markdown('<h1 style="text-align: center; color: green;">Chemical Dosage Prediction</h1>', unsafe_allow_html=True)

# Initialize session state
if "predictions_log" not in st.session_state:
    st.session_state.predictions_log = []  # Initialize log for storing predictions

# Tabs for Prediction & Graph
tab1, tab2 = st.tabs(["Prediction", "Prediction History"])

with tab1:
    # Upload Image Section
    st.header("Upload Wastewater Image", anchor="upload_image")

    uploaded_file = st.file_uploader("Upload Wastewater Image", type=["jpg", "png", "jpeg"])

    # Function to Extract Beaker Color using **Circle Detection + Masking**
    def extract_wastewater_color(image):
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=50, maxRadius=300)

        if circles is not None:
            circles = np.uint16(np.around(circles))[0, 0]  # Get first detected circle
            x, y, r = circles  # Get (x, y) center and radius

            # Create a circular mask to extract the liquid region
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r - 10, (255, 255, 255), thickness=-1)  # Subtract 10px to avoid edges

            # Apply the mask to extract only the inner liquid region
            masked_image = cv2.bitwise_and(img_np, img_np, mask=mask)

            # Crop the detected liquid area
            liquid_area = masked_image[y - r:y + r, x - r:x + r]

            # Compute the average color inside the detected wastewater area
            avg_hsv = np.mean(liquid_area, axis=(0, 1))

            # Draw the detected beaker with a bounding circle
            detected_img = img_np.copy()
            cv2.circle(detected_img, (x, y), r, (0, 255, 0), 4)  # Green Circle
            detected_img = Image.fromarray(detected_img)

            return avg_hsv[0], avg_hsv[1], avg_hsv[2], detected_img

        else:
            return None, None, None, img  # Return original image if no beaker detected

    # Input Section
    if uploaded_file is not None:
        hue, saturation, value, result_img = extract_wastewater_color(uploaded_file)

        if hue is not None:
            st.image(result_img, caption="Detected Wastewater Area", use_container_width=True)
            st.success(f"Detected Color - Hue: {hue:.2f}, Saturation: {saturation:.2f}, Value: {value:.2f}")
        else:
            st.error("Beaker not detected. Please use a clearer image.")

        # Prediction and Log Data
        flow_rate = st.number_input("Flow Rate (m³/h):", min_value=0.0, step=0.1)

        if st.button("Predict Chemical Dosage"):
            input_data = pd.DataFrame([[hue, saturation, value]], columns=feature_columns)
            prediction = model.predict(input_data)[0]

            percentage_ratio = (prediction / 500) * 100
            required_chemical_liters = (prediction / 500) * flow_rate * 1000
            required_chemical_cubic_meters = required_chemical_liters / 1000

            st.success(f"Chemical Required: {prediction:.2f} ml (per 500ml wastewater)")
            st.info(f"Dosage: {percentage_ratio:.2f}% of wastewater volume")
            st.warning(f"Estimated for {flow_rate} m³/h: {required_chemical_liters:.2f} L/h | {required_chemical_cubic_meters:.5f} m³/h")

            # Log Data to session_state (Keeps previous values)
            prediction_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Predicted Chemical (ml)": prediction,
                "Ratio (%)": percentage_ratio,
                "Chemical Volume (L/h)": required_chemical_liters,
                "Chemical Volume (m³/h)": required_chemical_cubic_meters
            }
            st.session_state.predictions_log.append(prediction_data)

    else:
        st.warning("Please upload an image to proceed.")

with tab2:
    st.header("Prediction History")

    # Always Display Previous Data Until Reset
    if len(st.session_state.predictions_log) > 0:
        df_log = pd.DataFrame(st.session_state.predictions_log)

        fig = px.line(
            df_log,
            x="Timestamp",
            y=["Chemical Volume (L/h)", "Chemical Volume (m³/h)"],
            markers=True,
            title="Chemical Dosage Over Time",
            labels={"Timestamp": "Time", "value": "Chemical Volume"},
            template="plotly_dark"
        )

        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

    # Button to Clear Graph History
    if st.button("Clear Graph History"):
        st.session_state.predictions_log = []  # Clear the history
        st.rerun()  # Refresh to reset the graph
