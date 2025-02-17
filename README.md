# OCR and Text Correction Project

This project extracts handwritten text from an image using the Google Cloud Vision API and then sends the text to the OpenAI API to correct grammar and context issues.

## Prerequisites

- **Python 3.9+**
- A Google Cloud project with the Vision API enabled.
- A service account JSON file for the Google Cloud Vision API.
- An OpenAI API key.

## Setup

1. **Clone the Repository**

```bash
git clone https://your-repository-url.git
cd your-repository-folder
```

2. **Create and Activate a Virtual Environment**

```bash
python -m venv .venv
```

Activate the virtual environment:

- macOS/Linux:

```bash
source .venv/bin/activate
```

- Windows:

```bash
.venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**

Create a `.env` file in the root directory of the project with the following content:

```dotenv
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service_account.json
```

Make sure the .env file is listed in your .gitignore file to keep your keys secure.

5. **Prepare Your Image**

Place the image you want to process (e.g., redacao-02.png) in the project directory or update the image path in main.py.

## Running the Project

To run the project, execute:

```bash
python main.py
```

The script will:

- Extract handwritten text from the provided image using Google Cloud Vision.
- Send the extracted text to the OpenAI API for correction.
- Print both the extracted text and the corrected text in the console.

## License

This project is licensed under the MIT License.
