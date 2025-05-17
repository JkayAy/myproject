# 🏥 Advanced Medical Diagnosis Platform

A comprehensive medical image analysis platform that leverages multiple AI models to provide detailed medical image analysis and diagnosis assistance. This platform combines deep learning with state-of-the-art language models to deliver comprehensive medical insights.

## 🌟 Features

- **Multi-Model Analysis**: Utilizes multiple AI models for comprehensive analysis:
  - Deep Learning (DenseNet121)
  - GPT-4 Vision
  - Claude 3
  - MedPaLM
  - Gemini

- **Patient Management**:
  - Medical history tracking
  - Current symptoms recording
  - Previous diagnoses logging
  - Medication management
  - Allergy tracking

- **Advanced Visualization**:
  - GradCAM heatmap overlays
  - Interactive probability charts
  - Comparative model analysis
  - Sentiment analysis visualization

- **User-Friendly Interface**:
  - Intuitive tab-based navigation
  - Real-time analysis results
  - Child-friendly explanations
  - Comprehensive evaluation dashboard

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)
- API keys for:
  - OpenAI (GPT-4 Vision)
  - Anthropic (Claude 3)
  - Google (Gemini)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-diagnosis-platform.git
cd medical-diagnosis-platform
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

## 📋 Usage

1. **Upload Medical Image**:
   - Click the upload button to select a medical image (CT, X-Ray, or MRI)
   - Supported formats: PNG, JPG, JPEG

2. **Enter Patient Information**:
   - Fill in the patient details in the sidebar
   - Include medical history, symptoms, and other relevant information

3. **Run Analysis**:
   - Choose from different analysis tabs
   - Click "Run Diagnosis" to start the analysis
   - View results and visualizations

4. **Review Results**:
   - Examine detailed analysis from each model
   - View probability distributions
   - Check GradCAM visualizations
   - Compare model outputs in the evaluation tab

## 🔧 Technical Details

### Model Architecture

- **Deep Learning Model**: DenseNet121 with custom classification head
- **Image Processing**: Standardized preprocessing pipeline
- **Analysis Pipeline**: Multi-stage analysis with different AI models
- **Evaluation System**: Comprehensive model comparison and evaluation

### Dependencies

- TensorFlow
- Streamlit
- OpenAI
- Anthropic
- Google Generative AI
- Plotly
- OpenCV
- NLTK
- Other requirements listed in `requirements.txt`

## ⚠️ Important Notes

- This platform is for assistance purposes only
- All diagnoses should be verified by qualified healthcare professionals
- Patient data should be handled according to relevant privacy regulations
- API usage is tracked for billing purposes

## 📊 Model Evaluation

The platform includes a comprehensive evaluation system that:
- Compares outputs from different models
- Analyzes confidence scores
- Evaluates response structure
- Provides detailed comparison reports

## 🔒 Security and Privacy

- Secure API key management
- Temporary file handling
- No permanent storage of patient data
- Compliance with medical data regulations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Ayodele James Kolawole

## 🙏 Acknowledgments

- OpenAI for GPT-4 Vision
- Anthropic for Claude 3
- Google for Gemini
- TensorFlow team for DenseNet121

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Disclaimer**: This platform is designed for assistance purposes only. Always consult qualified healthcare professionals for medical decisions.
