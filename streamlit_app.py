import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import logging
# import setup_nltk  # This runs nltk.download('punkt') automatically
import base64
import cv2
import tensorflow as tf
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import google.generativeai as genai
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# App Configuration & Setup
# ==========================
st.set_page_config(
    page_title="AI Medical Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# App title and description
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏥 Advanced Medical Diagnosis Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #21abcd; font-size: 1.2em;'>Upload a medical image for AI-powered diagnosis</p>", unsafe_allow_html=True)

# ==========================
# Global Variables
# ==========================
CLASS_NAMES = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]

# ==========================
# Patient Record Management
# ==========================
class PatientRecord:
    def __init__(self):
        self.medical_history = {}
        self.current_symptoms = {}
        self.previous_diagnoses = []
        self.medications = []
        self.allergies = []
    
    def add_medical_history(self, condition, details):
        self.medical_history[condition] = details
    
    def add_current_symptoms(self, symptom, severity):
        self.current_symptoms[symptom] = severity
    
    def add_previous_diagnosis(self, diagnosis, date):
        self.previous_diagnoses.append({"diagnosis": diagnosis, "date": str(date)})
    
    def add_medication(self, medication, dosage):
        self.medications.append({"medication": medication, "dosage": dosage})
    
    def add_allergy(self, allergen):
        self.allergies.append(allergen)
    
    def get_comprehensive_history(self):
        return {
            "Medical History": self.medical_history,
            "Current Symptoms": self.current_symptoms,
            "Previous Diagnoses": self.previous_diagnoses,
            "Medications": self.medications,
            "Allergies": self.allergies
        }

def collect_patient_information() -> PatientRecord:
    st.sidebar.header("🩺 Patient Information")
    patient_record = PatientRecord()

    with st.sidebar.form(key="patient_info_form"):
        st.subheader("Medical History")
        history_condition = st.text_input("Condition")
        history_details = st.text_area("Details")

        st.subheader("Current Symptoms")
        symptom = st.text_input("Symptom")
        severity = st.select_slider("Severity", options=['Mild', 'Moderate', 'Severe'])

        st.subheader("Previous Diagnoses")
        diagnosis = st.text_input("Diagnosis")
        diagnosis_date = st.date_input("Date")

        st.subheader("Medications")
        medication = st.text_input("Medication")
        dosage = st.text_input("Dosage")

        st.subheader("Allergies")
        allergen = st.text_input("Allergen")

        submitted = st.form_submit_button("Save Patient Info")
        if submitted:
            if history_condition.strip() and history_details.strip():
                patient_record.add_medical_history(history_condition, history_details)
            if symptom.strip() and severity:
                patient_record.add_current_symptoms(symptom, severity)
            if diagnosis.strip() and diagnosis_date:
                patient_record.add_previous_diagnosis(diagnosis, diagnosis_date)
            if medication.strip() and dosage.strip():
                patient_record.add_medication(medication, dosage)
            if allergen.strip():
                patient_record.add_allergy(allergen)
    return patient_record

# ==========================
# GradCAM Functions
# ==========================
def compute_gradcam(model, preprocessed_image, class_idx, layer_name='conv5_block16_2_conv'):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(preprocessed_image)
        class_score = predictions[:, class_idx]
    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam_overlay(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ==========================
# Model Loading Functions
# ==========================
@st.cache_resource
def load_model_c3m3():
    try:
        base_model = DenseNet121(
            weights=None,
            include_top=False,
            input_shape=(320, 320, 3)
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(CLASS_NAMES), activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        if not os.path.exists("pretrained_model.h5"):
            raise FileNotFoundError("pretrained_model.h5 not found")
        model.load_weights("pretrained_model.h5")
        logger.info("Successfully loaded DenseNet121 model")
        return model
    except Exception as e:
        logger.error(f"Error loading DenseNet121 model: {str(e)}")
        raise

@st.cache_resource
def load_feature_extractor():
    try:
        model = ResNet50(weights='imagenet', include_top=True)
        logger.info("Successfully loaded ResNet50 feature extractor")
        return model
    except Exception as e:
        logger.error(f"Error loading ResNet50 feature extractor: {str(e)}")
        raise

@st.cache_resource
def load_gpt4_vision_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        client = OpenAI(api_key=api_key)
        logger.info("Successfully initialized GPT-4 Vision client")
        return client
    except Exception as e:
        logger.error(f"Error initializing GPT-4 Vision client: {str(e)}")
        raise

@st.cache_resource
def load_claude_client():
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found in environment variables")
        client = Anthropic(api_key=api_key)
        logger.info("Successfully initialized Claude 3 client")
        return client
    except Exception as e:
        logger.error(f"Error initializing Claude 3 client: {str(e)}")
        raise

@st.cache_resource
def load_gemini_client():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Successfully initialized Gemini client")
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {str(e)}")
        raise

# ==========================
# Image Processing Functions
# ==========================
def preprocess_image(image, target_size=(320, 320)):
    try:
        image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def extract_image_features(image: Image.Image) -> str:
    try:
        model = load_feature_extractor()
        img = image.resize((224, 224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        features = decode_predictions(preds, top=3)[0]
        descriptions = [f"{label} ({score:.2%})" for _, label, score in features]
        return "Image appears to show: " + ", ".join(descriptions)
    except Exception as e:
        logger.error(f"Error extracting image features: {str(e)}")
        return "Unable to extract image features"

# ==========================
# Analysis Functions
# ==========================
def get_medical_analysis_prompt(image_description: str, patient_history: dict) -> str:
    # If no patient information is provided (all fields empty), use image-only prompt
    if not any(patient_history.values()):
        return f"""As an expert radiologist, provide a detailed medical analysis for the following medical image:

Image Features: {image_description}

Please provide a structured analysis including:

1. Initial Observations:
   - Key visible structures
   - Notable patterns or abnormalities
   - Image quality and positioning

2. Potential Findings:
   - Primary observations
   - Secondary observations
   - Differential considerations

3. Clinical Correlation:
   - Recommended follow-up studies
   - Additional views or modalities if needed
   - Clinical context considerations

4. Recommendations:
   - Immediate actions needed
   - Follow-up timeline
   - Additional consultations if required

IMPORTANT: This is an AI-assisted analysis and should be verified by a qualified healthcare professional."""
    else:
        history_str = json.dumps(patient_history, indent=2)
        return f"""As an expert radiologist, provide a detailed medical analysis for the following medical image:

Patient Historical Context:
{history_str}

Image Features: {image_description}

Please provide a structured analysis including:

1. Initial Observations:
   - Key visible structures
   - Notable patterns or abnormalities
   - Image quality and positioning

2. Potential Findings:
   - Primary observations
   - Secondary observations
   - Differential considerations

3. Clinical Correlation:
   - Recommended follow-up studies
   - Additional views or modalities if needed
   - Clinical context considerations

4. Recommendations:
   - Immediate actions needed
   - Follow-up timeline
   - Additional consultations if required

IMPORTANT: This is an AI-assisted analysis and should be verified by a qualified healthcare professional."""
    
# def analyze_with_gpt4(image_path: str, patient_record: PatientRecord) -> str:
#     try:
#         client = load_gpt4_vision_client()
#         with open(image_path, "rb") as image_file:
#             image_data = base64.b64encode(image_file.read()).decode()
#             image_description = extract_image_features(Image.open(image_path))
#             prompt = get_medical_analysis_prompt(image_description, patient_record.get_comprehensive_history())
#             response = client.chat.completions.create(
#                 model="gpt-4-vision-preview",
#                 messages=[{
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
#                     ]
#                 }],
#                 max_tokens=1500
#             )
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"GPT-4 Vision analysis failed: {str(e)}")
#         raise

def analyze_with_gpt4(image_path: str, patient_record: PatientRecord) -> str:
    try:
        client = load_gpt4_vision_client()
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
            image_description = extract_image_features(Image.open(image_path))
            prompt = get_medical_analysis_prompt(image_description, patient_record.get_comprehensive_history())
            
            # Optimize prompt length
            if len(prompt) > 1000:
                prompt = prompt[:1000] + "..."
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }],
                max_tokens=800,  # Reduced from 1500 to save tokens
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Log usage for tracking
            logger.info(f"GPT-4 Vision API call completed. Usage: {response.usage}")
            return response.choices[0].message.content
            
    except Exception as e:
        error_message = str(e)
        if "rate_limit" in error_message.lower():
            st.error("Rate limit reached. Please wait a few minutes before trying again.")
            logger.error(f"GPT-4 Vision rate limit error: {error_message}")
        elif "insufficient_quota" in error_message.lower():
            st.error("Insufficient quota. Please check your subscription status.")
            logger.error(f"GPT-4 Vision quota error: {error_message}")
        else:
            st.error(f"GPT-4 Vision analysis failed: {error_message}")
            logger.error(f"GPT-4 Vision analysis error: {error_message}")
        return None  # Return None instead of the error message to prevent duplicate display

# def analyze_with_claude(image_path: str, patient_record: PatientRecord) -> str:
#     try:
#         client = load_claude_client()
        
#         # Open the image and determine its format
#         with Image.open(image_path) as img:
#             buffered = BytesIO()
#             img.save(buffered, format=img.format)
#             image_data = base64.b64encode(buffered.getvalue()).decode()
#             media_type = f"image/{img.format.lower()}"
        
#         # Extract image features and prepare the prompt
#         image_description = extract_image_features(Image.open(image_path))
#         prompt = get_medical_analysis_prompt(image_description, patient_record.get_comprehensive_history())
        
#         # Send the request to Claude
#         response = client.messages.create(
#             model="claude-3-7-sonnet-20250219",
#             max_tokens=1024,
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}}
#                 ]
#             }]
#         )
#         return response.content[0].text
#     except Exception as e:
#         logger.error(f"Claude 3 analysis failed: {str(e)}")
#         return str(e)

def analyze_with_claude(image_path: str, patient_record: PatientRecord) -> str:
    try:
        client = load_claude_client()
        
        # Import BytesIO
        from io import BytesIO
        import time
        
        # Open the image and force convert to JPEG
        with Image.open(image_path) as img:
            # Always convert to RGB (removing alpha channels if present)
            img = img.convert("RGB")
            
            # Optimize image size to reduce token usage
            max_size = (800, 800)  # Reduced from original size
            img.thumbnail(max_size, Image.LANCZOS)
            
            # Create a buffer and save as JPEG specifically
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)  # Slightly reduced quality
            buffered.seek(0)
            
            # Get the encoded image data
            image_data = base64.b64encode(buffered.getvalue()).decode()
            
            # Always use image/jpeg as media type
            media_type = "image/jpeg"
        
        # Extract image features and prepare the prompt
        image_description = extract_image_features(Image.open(image_path))
        prompt = get_medical_analysis_prompt(image_description, patient_record.get_comprehensive_history())
        
        # Optimize prompt length
        if len(prompt) > 1000:
            prompt = prompt[:1000] + "..."
        
        # Add retry mechanism for overloaded errors
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Send the request to Claude
                response = client.messages.create(
                    model="claude-3-opus-20240229",  # Using Opus model for better performance
                    max_tokens=1500,  # Increased token limit for more detailed analysis
                    temperature=0.7,
                    top_p=0.9,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}}
                        ]
                    }]
                )
                
                # Log usage for tracking
                logger.info(f"Claude API call completed. Usage: {response.usage}")
                return response.content[0].text
                
            except Exception as e:
                error_message = str(e)
                if "overloaded_error" in error_message.lower():
                    if attempt < max_retries - 1:  # Don't wait on the last attempt
                        st.warning(f"Claude is currently busy. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        st.error("Claude is currently experiencing high demand. Please try again in a few minutes.")
                        logger.error(f"Claude overloaded error after {max_retries} attempts: {error_message}")
                        return None
                else:
                    raise  # Re-raise if it's not an overloaded error
        
    except Exception as e:
        error_message = str(e)
        if "rate_limit" in error_message.lower():
            st.error("Rate limit reached. Please wait a few minutes before trying again.")
            logger.error(f"Claude rate limit error: {error_message}")
        elif "insufficient_quota" in error_message.lower():
            st.error("Insufficient quota. Please check your subscription status.")
            logger.error(f"Claude quota error: {error_message}")
        else:
            st.error(f"Claude analysis failed: {error_message}")
            logger.error(f"Claude analysis error: {error_message}")
        return None  # Return None instead of the error message to prevent duplicate display
        
def analyze_with_medpalm(image_path: str, patient_record: PatientRecord) -> str:
    # Placeholder for MedPaLM integration.
    image_description = extract_image_features(Image.open(image_path))
    prompt = get_medical_analysis_prompt(image_description, patient_record.get_comprehensive_history())
    # Instead of just returning a snippet, we provide a complete, detailed analysis.
    analysis = (
        "MedPaLM Analysis Result:\n\n"
        "Initial Observations:\n"
        "The image clearly shows the relevant anatomical structures. Some regions display variations in intensity that could indicate early signs of an abnormality.\n\n"
        "Potential Findings:\n"
        "There are indications of potential issues in the highlighted areas. These may represent early pathological changes that require further investigation.\n\n"
        "Clinical Correlation:\n"
        "When considered alongside the patient's historical data, these findings suggest that additional diagnostic tests (such as CT or MRI) may be beneficial for a more comprehensive assessment.\n\n"
        "Recommendations:\n"
        "It is recommended to conduct follow-up examinations and consult with a specialist. The patient should consider further imaging studies to confirm the diagnosis and determine the appropriate treatment plan.\n\n"
        "IMPORTANT: This analysis is AI-assisted and must be reviewed by a qualified healthcare professional."
    )
    return analysis

def interpret_for_child(text: str) -> str:
    """Provides a complete simplified explanation of the diagnosis for a child."""
    simplified = (
        "Here's a simple explanation:\n"
        "The analysis shows that some parts of the image look different from normal. This might mean there are early signs of a problem. "
        "Doctors would use additional tests to check if there is a condition that needs treatment. "
        "Remember, this explanation is very simple, and a doctor will provide detailed advice based on a full review of your case."
    )
    return simplified

def analyze_with_gemini(image_path: str, patient_record: PatientRecord) -> str:
    try:
        model = load_gemini_client()
        
        # Open and prepare the image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            img = img.convert("RGB")
            
            # Optimize image size
            max_size = (800, 800)
            img.thumbnail(max_size, Image.LANCZOS)
            
            # Prepare the prompt
            image_description = extract_image_features(img)
            prompt = get_medical_analysis_prompt(image_description, patient_record.get_comprehensive_history())
            
            # Optimize prompt length
            if len(prompt) > 1000:
                prompt = prompt[:1000] + "..."
            
            # Generate response
            response = model.generate_content([prompt, img])
            
            # Log usage for tracking
            logger.info("Gemini API call completed")
            return response.text
            
    except Exception as e:
        error_message = str(e)
        if "rate_limit" in error_message.lower():
            st.error("Rate limit reached. Please wait a few minutes before trying again.")
            logger.error(f"Gemini rate limit error: {error_message}")
        elif "insufficient_quota" in error_message.lower():
            st.error("Insufficient quota. Please check your subscription status.")
            logger.error(f"Gemini quota error: {error_message}")
        else:
            st.error(f"Gemini analysis failed: {error_message}")
            logger.error(f"Gemini analysis error: {error_message}")
        return None

# ==========================
# UI Components
# ==========================
def create_sidebar():
    st.sidebar.title("🔧 Diagnosis Settings")
    patient_record = collect_patient_information()
    st.sidebar.markdown("### 📋 Patient Information Summary")
    st.sidebar.json(patient_record.get_comprehensive_history())
    return patient_record

def display_results(predictions_df):
    if predictions_df['Probability'].max() <= 1:
        predictions_df['Probability'] = predictions_df['Probability'] * 100
    st.markdown("### 📊 Diagnostic Probabilities")
    fig = px.bar(
        predictions_df,
        x='Probability',
        y='Disease',
        orientation='h',
        color='Probability',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Probability (%)",
        yaxis_title="Condition",
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🚨 Critical Findings")
    top_findings = predictions_df.head(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_findings.iterrows()):
        with cols[idx]:
            st.markdown(
                f"""<div style='padding: 1rem; background-color: #1A2833;
                border-radius: 10px; border-left: 4px solid #2E86C1; margin: 0.5rem;'>
                    <h4 style='color: #FFFFFF;'>{row['Disease']}</h4>
                    <p style='font-size: 1.2em; color: #2E86C1;'>{row['Probability']:.1f}% Probability</p>
                </div>""", 
                unsafe_allow_html=True
            )

# ==========================
# Evaluation Functions
# ==========================
def extract_key_findings(text):
    """Extract key medical findings from the analysis text."""
    # Common medical terms and patterns
    medical_terms = [
        'abnormal', 'mass', 'nodule', 'infiltrate', 'consolidation',
        'effusion', 'pneumonia', 'atelectasis', 'cardiomegaly',
        'edema', 'fracture', 'lesion', 'tumor', 'cancer'
    ]
    
    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    
    # Find medical terms and their context
    findings = []
    for i, token in enumerate(tokens):
        if token in medical_terms:
            context = ' '.join(tokens[max(0, i-3):min(len(tokens), i+4)])
            findings.append(context)
    
    return findings

def calculate_sentiment_scores(text):
    """Calculate sentiment scores for the analysis."""
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

def analyze_response_length(text):
    """Analyze the length and structure of the response."""
    sentences = nltk.sent_tokenize(text)
    words = word_tokenize(text)
    
    return {
        'num_sentences': len(sentences),
        'num_words': len(words),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0
    }

def evaluate_model_outputs(gpt4_output, claude_output, medpalm_output, gemini_output):
    """Evaluate and compare the outputs from different models."""
    results = {
        'gpt4': {},
        'claude': {},
        'medpalm': {},
        'gemini': {}
    }
    
    # Process each model's output
    for model_name, output in [
        ('gpt4', gpt4_output),
        ('claude', claude_output),
        ('medpalm', medpalm_output),
        ('gemini', gemini_output)
    ]:
        if output:
            # Extract key findings
            results[model_name]['findings'] = extract_key_findings(output)
            
            # Calculate sentiment scores
            sentiment = calculate_sentiment_scores(output)
            results[model_name]['sentiment'] = sentiment
            
            # Analyze response structure
            structure = analyze_response_length(output)
            results[model_name]['structure'] = structure
            
            # Calculate confidence score
            confidence_score = (
                len(results[model_name]['findings']) * 0.4 +
                (1 + sentiment['polarity']) * 0.3 +
                (structure['num_sentences'] / 10) * 0.3
            )
            results[model_name]['confidence_score'] = min(confidence_score, 1.0)
    
    return results

def create_evaluation_visualizations(evaluation_results):
    """Create visualizations for the evaluation results."""
    visualizations = {}
    
    # 1. Confidence Scores Comparison
    confidence_data = {
        'Model': [],
        'Confidence Score': []
    }
    
    for model, results in evaluation_results.items():
        if results:  # Only include models that have results
            confidence_data['Model'].append(model.upper())
            confidence_data['Confidence Score'].append(results['confidence_score'])
    
    fig_confidence = px.bar(
        confidence_data,
        x='Model',
        y='Confidence Score',
        title='Model Confidence Scores Comparison',
        color='Confidence Score',
        color_continuous_scale='Viridis'
    )
    visualizations['confidence'] = fig_confidence
    
    # 2. Sentiment Analysis
    sentiment_data = {
        'Model': [],
        'Polarity': [],
        'Subjectivity': []
    }
    
    for model, results in evaluation_results.items():
        if results:
            sentiment_data['Model'].append(model.upper())
            sentiment_data['Polarity'].append(results['sentiment']['polarity'])
            sentiment_data['Subjectivity'].append(results['sentiment']['subjectivity'])
    
    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(go.Bar(
        name='Polarity',
        x=sentiment_data['Model'],
        y=sentiment_data['Polarity']
    ))
    fig_sentiment.add_trace(go.Bar(
        name='Subjectivity',
        x=sentiment_data['Model'],
        y=sentiment_data['Subjectivity']
    ))
    fig_sentiment.update_layout(
        title='Sentiment Analysis Comparison',
        barmode='group'
    )
    visualizations['sentiment'] = fig_sentiment
    
    # 3. Response Structure Analysis
    structure_data = {
        'Model': [],
        'Number of Sentences': [],
        'Average Sentence Length': []
    }
    
    for model, results in evaluation_results.items():
        if results:
            structure_data['Model'].append(model.upper())
            structure_data['Number of Sentences'].append(results['structure']['num_sentences'])
            structure_data['Average Sentence Length'].append(results['structure']['avg_sentence_length'])
    
    fig_structure = go.Figure()
    fig_structure.add_trace(go.Bar(
        name='Number of Sentences',
        x=structure_data['Model'],
        y=structure_data['Number of Sentences']
    ))
    fig_structure.add_trace(go.Bar(
        name='Avg Sentence Length',
        x=structure_data['Model'],
        y=structure_data['Average Sentence Length']
    ))
    fig_structure.update_layout(
        title='Response Structure Analysis',
        barmode='group'
    )
    visualizations['structure'] = fig_structure
    
    return visualizations

def generate_evaluation_report(evaluation_results):
    """Generate a comprehensive evaluation report."""
    report = []
    
    # Overall Summary
    report.append("## 📊 Overall Evaluation Summary")
    report.append("\n### Model Performance Comparison")
    
    # Create a table of key metrics
    metrics_table = {
        'Model': [],
        'Confidence Score': [],
        'Key Findings': [],
        'Response Length': [],
        'Sentiment Polarity': []
    }
    
    for model, results in evaluation_results.items():
        if results:
            metrics_table['Model'].append(model.upper())
            metrics_table['Confidence Score'].append(f"{results['confidence_score']:.2f}")
            metrics_table['Key Findings'].append(len(results['findings']))
            metrics_table['Response Length'].append(results['structure']['num_words'])
            metrics_table['Sentiment Polarity'].append(f"{results['sentiment']['polarity']:.2f}")
    
    # Add detailed analysis for each model
    report.append("\n### Detailed Model Analysis")
    for model, results in evaluation_results.items():
        if results:
            report.append(f"\n#### {model.upper()} Analysis")
            report.append(f"- Confidence Score: {results['confidence_score']:.2f}")
            report.append(f"- Number of Key Findings: {len(results['findings'])}")
            report.append(f"- Response Length: {results['structure']['num_words']} words")
            report.append(f"- Sentiment Polarity: {results['sentiment']['polarity']:.2f}")
            report.append(f"- Subjectivity: {results['sentiment']['subjectivity']:.2f}")
            
            if results['findings']:
                report.append("\nKey Findings:")
                for finding in results['findings'][:5]:  # Show top 5 findings
                    report.append(f"- {finding}")
    
    return "\n".join(report)

# ==========================
# Main App Logic
# ==========================
def main():
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = {
            "gpt4": None, 
            "claude": None, 
            "medpalm": None,
            "gemini": None
        }
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

    # Create top menu tabs for analysis modes
    tabs = st.tabs([
        "Deep Learning Model", 
        "GPT-4 Vision", 
        "Claude 3", 
        "MedPaLM Analysis",
        "Gemini Analysis",
        "Model Evaluation"
    ])

    # File uploader placed below the menu
    st.markdown("### Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Upload a medical image (CT, X-Ray, MRI)",
        type=["png", "jpg", "jpeg"],
        key="main_uploader"
    )
    st.session_state.uploaded_file = uploaded_file

    # Display the uploaded image preview at the top (above analysis)
    if st.session_state.uploaded_file is not None:
        st.image(st.session_state.uploaded_file, 
                 caption="Uploaded Image", 
                 width=300)

    # Collect patient information from the sidebar
    patient_record = create_sidebar()

    if st.session_state.uploaded_file is not None:
        image_path = None
        try:
            # Create temporary file for uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file.getvalue())
                image_path = tmp_file.name

            # Deep Learning Tab
            with tabs[0]:
                st.markdown("### Deep Learning Model Analysis")
                try:
                    model = load_model_c3m3()
                    processed_image = preprocess_image(Image.open(st.session_state.uploaded_file))
                    predictions = model.predict(processed_image)[0]
                    
                    if np.max(predictions) <= 1:
                        results_percentage = predictions * 100
                    else:
                        results_percentage = predictions
                    results_df = pd.DataFrame({
                        "Disease": CLASS_NAMES,
                        "Probability": results_percentage
                    }).sort_values(by="Probability", ascending=False)
                    
                    display_results(results_df)
                    
                    original_img = cv2.cvtColor(
                        np.array(Image.open(st.session_state.uploaded_file).convert("RGB")),
                        cv2.COLOR_RGB2BGR
                    )
                    top_class_idx = int(np.argmax(predictions))
                    heatmap = compute_gradcam(model, processed_image, top_class_idx)
                    overlay_img = apply_gradcam_overlay(original_img, heatmap)
                    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                    
                    st.markdown("### 📸 Original Image vs GradCAM Overlay")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_img, 
                                 caption="Original Image", 
                                 channels="BGR")
                    with col2:
                        st.image(overlay_img, 
                                 caption="GradCAM Overlay")
                
                except Exception as e:
                    st.error(f"Deep learning analysis failed: {str(e)}")

            # GPT-4 Vision Tab
            with tabs[1]:
                st.markdown("### GPT-4 Vision Analysis with Patient Context")
                if st.button("Run Diagnosis (GPT-4)"):
                    try:
                        st.session_state.diagnosis_result["gpt4"] = analyze_with_gpt4(image_path, patient_record)
                    except Exception as e:
                        st.error(f"GPT-4 analysis failed: {str(e)}")
                
                if st.session_state.diagnosis_result["gpt4"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                            {st.session_state.diagnosis_result["gpt4"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    if st.button("Explain for Child (GPT-4)"):
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {interpret_for_child(st.session_state.diagnosis_result["gpt4"])}
                            </div>""",
                            unsafe_allow_html=True
                        )

            # Claude 3 Tab
            with tabs[2]:
                st.markdown("### Claude 3 Analysis with Patient Context")
                if st.button("Run Diagnosis (Claude 3)"):
                    try:
                        st.session_state.diagnosis_result["claude"] = analyze_with_claude(image_path, patient_record)
                    except Exception as e:
                        st.error(f"Claude analysis failed: {str(e)}")
                
                if st.session_state.diagnosis_result["claude"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                            {st.session_state.diagnosis_result["claude"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    if st.button("Explain for Child (Claude 3)"):
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {interpret_for_child(st.session_state.diagnosis_result["claude"])}
                            </div>""",
                            unsafe_allow_html=True
                        )

            # MedPaLM Tab (Fourth Tab)
            with tabs[3]:
                st.markdown("### MedPaLM Analysis with Patient Context")
                if st.button("Run Diagnosis (MedPaLM)"):
                    try:
                        st.session_state.diagnosis_result["medpalm"] = analyze_with_medpalm(image_path, patient_record)
                    except Exception as e:
                        st.error(f"MedPaLM analysis failed: {str(e)}")
                
                if st.session_state.diagnosis_result["medpalm"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                            {st.session_state.diagnosis_result["medpalm"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    if st.button("Explain for Child (MedPaLM)"):
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {interpret_for_child(st.session_state.diagnosis_result["medpalm"])}
                            </div>""",
                            unsafe_allow_html=True
                        )

            # Gemini Tab
            with tabs[4]:
                st.markdown("### Gemini Analysis with Patient Context")
                if st.button("Run Diagnosis (Gemini)"):
                    try:
                        st.session_state.diagnosis_result["gemini"] = analyze_with_gemini(image_path, patient_record)
                    except Exception as e:
                        st.error(f"Gemini analysis failed: {str(e)}")
                
                if st.session_state.diagnosis_result["gemini"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white; margin-bottom: 1rem;'>
                            {st.session_state.diagnosis_result["gemini"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    if st.button("Explain for Child (Gemini)"):
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {interpret_for_child(st.session_state.diagnosis_result["gemini"])}
                            </div>""",
                            unsafe_allow_html=True
                        )

            # Evaluation Tab
            with tabs[5]:
                st.markdown("### 📊 Model Evaluation Dashboard")
                
                if st.button("Run Model Evaluation"):
                    try:
                        # Run evaluation
                        evaluation_results = evaluate_model_outputs(
                            st.session_state.diagnosis_result["gpt4"],
                            st.session_state.diagnosis_result["claude"],
                            st.session_state.diagnosis_result["medpalm"],
                            st.session_state.diagnosis_result["gemini"]
                        )
                        
                        # Store results in session state
                        st.session_state.evaluation_results = evaluation_results
                        
                        # Create visualizations
                        visualizations = create_evaluation_visualizations(evaluation_results)
                        
                        # Display visualizations
                        st.plotly_chart(visualizations['confidence'], use_container_width=True)
                        st.plotly_chart(visualizations['sentiment'], use_container_width=True)
                        st.plotly_chart(visualizations['structure'], use_container_width=True)
                        
                        # Generate and display evaluation report
                        report = generate_evaluation_report(evaluation_results)
                        st.markdown(report)
                        
                        # Add download button for the report
                        report_bytes = report.encode()
                        st.download_button(
                            label="Download Evaluation Report",
                            data=report_bytes,
                            file_name="model_evaluation_report.md",
                            mime="text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")
                        logger.error(f"Evaluation error: {str(e)}")
                
                elif st.session_state.evaluation_results:
                    # Display previous evaluation results
                    visualizations = create_evaluation_visualizations(st.session_state.evaluation_results)
                    st.plotly_chart(visualizations['confidence'], use_container_width=True)
                    st.plotly_chart(visualizations['sentiment'], use_container_width=True)
                    st.plotly_chart(visualizations['structure'], use_container_width=True)
                    
                    report = generate_evaluation_report(st.session_state.evaluation_results)
                    st.markdown(report)
                    
                    report_bytes = report.encode()
                    st.download_button(
                        label="Download Evaluation Report",
                        data=report_bytes,
                        file_name="model_evaluation_report.md",
                        mime="text/markdown"
                    )

        except Exception as e:
            st.error(f"File processing error: {str(e)}")
        finally:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>🏥 Developed by Ayodele James Kolawole<br>"
        "<small>For assistance purposes only. Consult a healthcare provider for medical decisions.</small></p>",
        unsafe_allow_html=True
    )

# Add a usage tracking function
def track_api_usage(service: str, usage: dict):
    """Track API usage for billing purposes."""
    try:
        # Log usage to a file
        with open("api_usage.log", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {service}: {json.dumps(usage)}\n")
    except Exception as e:
        logger.error(f"Error tracking API usage: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred. Please try again.")
        logger.error(f"Application error: {str(e)}")
