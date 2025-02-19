import os
import io
import logging
from datetime import datetime
from dotenv import load_dotenv
import openai
from google.cloud import vision
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

def setup_logging():
    """Configure logging with timestamp, level, and message format"""
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_directory}/api_processing_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_handwritten_text(image_path):
    """Extracts handwritten text from the image using Google Cloud Vision API"""
    try:
        client = vision.ImageAnnotatorClient()
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API Error: {response.error.message}")

        if response.text_annotations:
            return response.text_annotations[0].description
        return ""

    except Exception as e:
        logging.error(f"Error in text extraction: {str(e)}", exc_info=True)
        raise


def correct_text_with_openai(text):
    """Corrects text using OpenAI"""
    try:
        prompt = (
            "A partir dessa redação, existem falhas de gramática e de contexto, "
            "me retorne a redação corrigindo essas falhas que não deveriam existir, "
            "a redação deve estar completa e com contexto re-estabelecido:\n\n"
            f"{text}"
        )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Iniciar correção da redação"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in text correction: {str(e)}", exc_info=True)
        raise


def evaluate_competencies(text, aux_prompt=''):
    """Evaluates ENEM competencies and provides scores"""
    try:
        base_prompt = f"""
        Você é uma Inteligência Artificial especializada em avaliação de redações do ENEM. Seu nome é Houzel.
        Você é paciente e gentil ao avaliar redações, e sempre fornece feedback construtivo para ajudar os estudantes a melhorar.
        Avalie o seguinte texto de acordo com as competências do ENEM, dando uma análise detalhada 
        para cada competência e justificando sua avaliação:

        Competência 1: Demonstrar domínio da modalidade escrita formal da língua portuguesa.
        Competência 2: Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento.
        Competência 3: Selecionar, relacionar, organizar e interpretar informações em defesa de um ponto de vista.
        Competência 4: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.
        Competência 5: Elaborar proposta de intervenção para o problema abordado que respeite os direitos humanos.

        Texto para avaliação:
        {text}

        Para cada competência, forneça uma análise detalhada e justificada.
        """

        # Append auxiliary prompt if provided
        prompt_competencies = base_prompt + ('\n\n' + aux_prompt if aux_prompt else '')

        response_competencies = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um avaliador especializado em redações do ENEM."},
                {"role": "user", "content": prompt_competencies}
            ]
        )

        competency_evaluation = response_competencies.choices[0].message.content

        prompt_score = f"""
        Com base na avaliação detalhada das competências acima, atribua uma nota final de 0 a 1000 
        para esta redação, justificando a pontuação com base nas competências avaliadas.

        Avaliação das competências:
        {competency_evaluation}

        Por favor, forneça:
        1. A nota final (0-1000)
        2. Uma breve justificativa da nota atribuída
        """

        response_score = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um avaliador especializado em redações do ENEM."},
                {"role": "user", "content": prompt_score}
            ]
        )

        final_score = response_score.choices[0].message.content
        return {
            "competency_evaluation": competency_evaluation,
            "final_score": final_score
        }

    except Exception as e:
        logging.error(f"Error in evaluation process: {str(e)}", exc_info=True)
        raise

def detect_ai_generated(text):
    """Detects if the text was likely generated by AI"""
    try:
        prompt = f"""
        Analise cuidadosamente o texto abaixo e determine se ele foi provavelmente escrito por uma IA ou por um humano.
        Considere aspectos como:
        - Naturalidade da linguagem
        - Padrões de escrita
        - Erros e imperfeições naturais
        - Fluxo de ideias
        - Marcas de subjetividade

        Texto para análise:
        {text}

        Forneça:
        1. Sua conclusão (AI-GENERATED ou HUMAN-WRITTEN)
        2. Nível de confiança (0-100%)
        3. Justificativa detalhada da sua análise
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um especialista em detectar textos gerados por IA."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in AI detection: {str(e)}", exc_info=True)
        raise


@app.route('/evaluate', methods=['POST'])
def evaluate_essay():
    """API endpoint to evaluate multiple essay images"""
    try:
        # Check if any images were uploaded
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        # Get list of all uploaded files
        files = request.files.getlist('images')
        # Get auxiliary prompt if provided
        aux_prompt = request.form.get('aux_prompt', '')

        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected files'}), 400

        # Process each image and collect text
        extracted_texts = []
        temp_files = []

        for file in files:
            if not allowed_file(file.filename):
                # Clean up any saved files before returning error
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            temp_files.append(filepath)

            # Extract text from each image
            try:
                text = extract_handwritten_text(filepath)
                extracted_texts.append(text)
            except Exception as e:
                # Clean up on extraction error
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                raise e

        # Combine all extracted text with double newlines between each
        combined_text = "\n\n".join(extracted_texts)
        logging.info(f"Combined text from {len(files)} images")

        # Process the combined text
        corrected_text = correct_text_with_openai(combined_text)

        # Check if AI detection is requested
        ai_detection = request.form.get('ai_detection', '').lower() == 'true'

        response_data = {}

        if ai_detection:
            ai_detection_result = detect_ai_generated(corrected_text)
            response_data['ai_detection'] = ai_detection_result

        evaluation_results = evaluate_competencies(corrected_text, aux_prompt)
        response_data.update(evaluation_results)

        # Clean up temporary files
        for filepath in temp_files:
            if os.path.exists(filepath):
                os.remove(filepath)

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_logging()
    app.run(debug=True)