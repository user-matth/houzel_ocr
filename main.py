import os
import io
from dotenv import load_dotenv
import openai
from google.cloud import vision

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def extract_handwritten_text(image_path):
    """
    Extracts handwritten text from the image at 'image_path' using the Google Cloud Vision API.
    Returns the recognized text.
    """
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    annotations = response.text_annotations

    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    if annotations:
        return annotations[0].description
    else:
        return ""

def correct_text_with_openai(text):
    """
    Sends the extracted text to OpenAI for grammar and context corrections.
    Returns the corrected text.
    """
    prompt = (
        "A partir dessa redação, existem falhas de gramática e de contexto, "
        "me retorne a redação corrigindo essas falhas que não deveriam existir, "
        "a redação deve estar completa e com contexto re-estabelecido:\n\n"
        f"{text}"
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Iniciar correção da redação"},
            {"role": "user", "content": prompt}
        ]
    )

    corrected_text = response.choices[0].message.content
    return corrected_text

if __name__ == "__main__":
    # Example usage: replace with the path to your image
    image_path = "redacao-02.png"

    extracted_text = extract_handwritten_text(image_path)
    print("Texto Extraído:\n")
    print(extracted_text)
    print("-" * 80)

    corrected_text = correct_text_with_openai(extracted_text)
    print("Texto Corrigido pela OpenAI:\n")
    print(corrected_text)