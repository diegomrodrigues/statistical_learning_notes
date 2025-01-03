import os
import time
import google.generativeai as genai

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "00. Prompts", "Topics.md")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Model configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)

def upload_to_gemini(path):
    print(f"Uploading: {path}")
    return genai.upload_file(path, mime_type="application/pdf")

def wait_for_file_active(file):
    print("Waiting for file processing...", end="", flush=True)
    while True:
        file = genai.get_file(file.name)
        if file.state.name == "ACTIVE":
            break
        elif file.state.name != "PROCESSING":
            raise Exception(f"File {file.name} failed to process")
        print(".", end="", flush=True)
        time.sleep(5)
    print("\nFile ready")

def create_chat_session(prompt_file):
    """Create a chat session with system prompt."""
    # Read prompt
    with open(prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    # Create model with system instruction
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=GENERATION_CONFIG,
        system_instruction=system_prompt
    )
    
    return model.start_chat()

def process_pdf(chat, pdf_path):
    print(f"\nProcessing PDF: {os.path.basename(pdf_path)}")
    
    # Upload and wait for file
    uploaded_file = upload_to_gemini(pdf_path)
    wait_for_file_active(uploaded_file)
    
    try:
        # Send the file to the chat session
        response = chat.send_message({
            "role": "user",
            "parts": [uploaded_file],
        })
                
        # Save output in the same directory as the PDF
        output_file = os.path.join(os.path.dirname(pdf_path), "topics.md")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(response.text)
        print(f"✓ Topics saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

def get_pdf_files():
    pdf_files = []
    # Walk through all directories in the root
    for root, _, files in os.walk(BASE_DIR):
        # Skip "00." directories
        if "00." in root:
            continue
        
        # Add all PDFs found
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def main():
    # Initialize Gemini
    init_gemini()
    
    # Get all PDF files
    pdf_files = get_pdf_files()
    if not pdf_files:
        print("No PDF files found!")
        return
    
    # Create single chat session for all PDFs
    chat = create_chat_session(PROMPT_FILE)
    
    # Process each PDF using the same chat session
    for pdf_file in pdf_files:
        try:
            process_pdf(chat, pdf_file)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue

if __name__ == "__main__":
    main()
