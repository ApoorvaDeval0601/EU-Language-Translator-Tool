# Import necessary libraries
import gradio as gr
import joblib
import re
import cv2
import pytesseract
import speech_recognition as sr
import numpy as np
from transformers import pipeline
import warnings
import os

# --- 1. CONFIGURATION & MODEL LOADING ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the path to the local language identification model
model_path = "language_detection_model.joblib"
pipe_lr = None

# Check if the local model file exists and load it
if os.path.exists(model_path):
    print(f"Loading local Language ID model ({model_path})...")
    pipe_lr = joblib.load(model_path)
    print("Language ID model loaded.")
else:
    # Print an error if the model file is not found
    print(f"ERROR: '{model_path}' not found.")
    print("Please make sure your trained model file is in the same folder.")

# --- Load Translator Models (FROM HUB) ---

# A dictionary to cache loaded translation models to avoid reloading
translator_cache = {}

# Maps the language names from the local model to the Helsinki-NLP model codes
LANG_CODE_MAP = {
    'English': 'en', 'German': 'de', 'French': 'fr', 'Spanish': 'es',
    'Bulgarian': 'bg', 'Czech': 'cs', 'Danish': 'da', 'Finnish': 'fi',
    'Hungarian': 'hu', 'Latvian': 'lv', 'Dutch': 'nl', 'Polish': 'pl',
    'Portuguese': 'pt', 'Romanian': 'ro', 'Slovenian': 'sl', 'Swedish': 'sv' 
}

def get_translator(language_name):
    """
    Loads a translator pipeline from Hugging Face Hub.
    Uses a cache to avoid re-loading the same model.
    """
    # If the text is already English, no translation is needed
    if language_name == 'English':
        return "English" 
        
    # Return the model from the cache if it's already loaded
    if language_name in translator_cache:
        return translator_cache[language_name] 
    
    # If the language is not in our map, we can't translate it
    if language_name not in LANG_CODE_MAP:
        return None 
        
    # Get the language code and construct the model name
    lang_code = LANG_CODE_MAP[language_name]
    model_name = f"Helsinki-NLP/opus-mt-{lang_code}-en"
    
    try:
        # Load the translation pipeline from Hugging Face
        print(f"Loading '{model_name}' from Hugging Face Hub... (This happens once per language)")
        translator = pipeline("translation", model=model_name)
        # Store the loaded model in the cache
        translator_cache[language_name] = translator 
        print("Model loaded and cached.")
        return translator
    except Exception as e:
        # Handle errors during model loading
        print(f"Error loading translator: {e}")
        return None

# --- 2. HELPER FUNCTIONS (Cleaning, STT, OCR) ---

def clean_text_for_id(text):
    """Cleans text to prepare it for the Language ID model."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def ocr_image(img_array):
    """Extracts text from an image (numpy array from Gradio)."""
    if img_array is None: return ""
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to binarize the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Use Pytesseract to perform OCR on the processed image
        text = pytesseract.image_to_string(thresh)
        return text
    except Exception as e:
        return f"[OCR Error: {e}]"

def stt_audio(audio_tuple):
    """Transcribes audio from a Gradio audio component tuple."""
    if audio_tuple is None: return ""
    sample_rate, audio_data = audio_tuple
    
    # Ensure audio data is in 16-bit integer format for SpeechRecognition
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
        
    # Convert numpy array to raw bytes
    audio_bytes = audio_data.tobytes()
    r = sr.Recognizer()
    try:
        # Create an AudioData object
        audio_source = sr.AudioData(audio_bytes, sample_rate, 2) # 2 bytes per sample (int16)
        print("Recognizing audio...")
        # Use Google's Speech Recognition API to transcribe
        text = r.recognize_google(audio_source)
        return text
    except Exception as e:
        return f"[STT Error: {e}]"

# --- 3. MAIN GRADIO FUNCTION ---

def identify_and_translate(text_input, image_input, voice_input, progress=gr.Progress(track_tqdm=True)):
    """
    This is the core function for the Gradio interface.
    It takes all possible inputs and returns all outputs.
    """
    
    raw_text = ""
    source = ""
    
    # 1. Determine which input was provided and get the raw text
    progress(0.1, desc="Processing Input...")
    if text_input:
        raw_text = text_input
        source = "Text"
    elif image_input is not None:
        raw_text = ocr_image(image_input)
        source = "Image"
    elif voice_input is not None:
        raw_text = stt_audio(voice_input)
        source = "Voice"
    
    # Handle cases with no valid input
    if not raw_text or not raw_text.strip():
        return "No input provided.", "N/A", "N/A"
        
    # 2. Identify the language using the local model
    progress(0.4, desc="Identifying Language...")
    cleaned_text = clean_text_for_id(raw_text)
    
    # Handle cases where cleaning removes all text
    if not cleaned_text:
         return "(Input was only symbols or numbers)", "N/A", raw_text
         
    # Check if the language ID model was loaded correctly
    if pipe_lr is None:
        return "ERROR: Language ID model not loaded.", "N/A", raw_text
        
    # Predict the language
    lang_id = pipe_lr.predict([cleaned_text])[0]
    
    # 3. Translate the text using the Hugging Face model
    progress(0.7, desc=f"Loading '{lang_id}' translator...")
    translation = ""
    # Load the translator (or get it from cache)
    translator = get_translator(lang_id) 
    
    if translator == "English":
        # No translation needed
        translation = "[Text is already in English]"
    elif translator:
        try:
            progress(0.9, desc=f"Translating from {lang_id}...")
            # Use the original raw text for translation
            result = translator(raw_text, max_length=512)
            translation = result[0]['translation_text']
        except Exception as e:
            translation = f"[Translation Error: {e}]"
    else:
        # Handle languages with no available model
        translation = f"[No translator model found for {lang_id}]"
        
    # Return the identified language, the translation, and the source text
    return lang_id, translation, f"Input from {source}:\n\n{raw_text}"

# --- 4. BUILD AND LAUNCH THE UI ---

# Create the Gradio interface using the Blocks layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Title and description
    gr.Markdown(
        """
        # Language ID & Translation Tool 
        This tool identifies the language (Text, Image, or Voice) and then uses a
        Helsinki-NLP model from Hugging Face to translate it to English.
        """
    )
    
    # Create tabs for the three input methods
    with gr.Tabs():
        with gr.TabItem("Text Input"):
            text_in = gr.Textbox(lines=7, label="Type Text Here", placeholder="Bonjour, comment ça va?")
        with gr.TabItem("Image Input"):
            image_in = gr.Image(type="numpy", label="Upload an Image with Text")
        with gr.TabItem("Voice Input"):
            voice_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Audio")
    
    # Create the submit button
    submit_btn = gr.Button("Process Input", variant="primary")
    
    # Add a separator
    gr.Markdown("---")
    gr.Markdown("### Results")
    
    # Create the output components
    lang_out = gr.Textbox(label="Identified Language", interactive=False, lines=1)   
    
    with gr.Row():
        processed_text_out = gr.Textbox(label="Text Detected from Source", interactive=False, lines=8)
        trans_out = gr.Textbox(label="English Translation", interactive=False, lines=8)
    
    # Link the submit button to the main function
    submit_btn.click(
        fn=identify_and_translate,
        inputs=[text_in, image_in, voice_in],
        outputs=[lang_out, trans_out, processed_text_out]
    )

# Standard Python entry point to run the app
if __name__ == "__main__":
    # Check again if the local model failed to load and warn the user
    if pipe_lr is None:
        print("\n--- WARNING: GRADIO WILL LAUNCH, BUT PREDICTIONS WILL FAIL ---")
        print("--- Please ensure 'language_detection_model.joblib' is in the correct folder. ---")
    
    # Launch the Gradio app
    print("\nLaunching Gradio UI...")
    demo.launch(share=True)