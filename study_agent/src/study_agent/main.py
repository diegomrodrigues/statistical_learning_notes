#!/usr/bin/env python
import os
import warnings
from datetime import datetime
import traceback
from pathlib import Path
from typing import List, Dict
import time
from functools import wraps
import re
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from crew import StudyAgent
from crewai import LLM
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Safety settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

os.environ["GOOGLE_API_KEY"] = "AIzaSyBrDXc2PAh7QNj1lT6IHLEQ-AJvMqImisI"

BASE_DIR = Path("/Workspace/Users/diego.rodrigues@stonex.com/statistical_learning_notes")

TARGET_FOLDERS = [
    "08. Random Forests"
]


# Configure Gemini API key before any operations
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def retry_on_error(max_retries=3):
    """Decorator to retry a function on error with logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt failed
                        error_msg = f"""
Time: {datetime.now()}
Function: {func.__name__}
Error: {str(e)}
Stack Trace:
{traceback.format_exc()}
----------------------------------------
"""
                        error_file = BASE_DIR / "errors.txt"
                        with open(error_file, 'a', encoding='utf-8') as f:
                            f.write(error_msg)
                        print(f"❌ Failed after {max_retries} attempts: {func.__name__}")
                        raise
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator

def get_input_directories(base_dir):
    """Get all valid input directories from the base directory."""
    input_dirs = []
    for item in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, item)
        # Skip if not a directory or starts with "00."
        if not os.path.isdir(dir_path) or item.startswith("00."):
            continue

        # Skip if target folders is specified
        if len(TARGET_FOLDERS) > 0:
            if not item in TARGET_FOLDERS:
                continue

        # Skip if no topics.md found
        if not any(f.lower() == "topics.md" for f in os.listdir(dir_path)):
            continue
        # Convert to Path object before adding to list
        input_dirs.append(Path(dir_path))
    return input_dirs


def get_pdf_files(directory: Path) -> List[Path]:
    """Get all PDF files in the directory."""
    return list(directory.glob("*.pdf"))

def read_topics_file(directory: Path) -> str:
    """Read and find the topics.md file in the given directory."""
    topics_file = directory / "topics.md"
    if topics_file.exists():
        return topics_file.read_text(encoding='utf-8')
    raise FileNotFoundError("topics.md not found in the specified directory")

def get_topics_dict(topics_content: str) -> Dict[str, List[str]]:
    """Convert topics content to dictionary using Gemini."""
    json_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    topics_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=json_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""Convert the topics into a JSON format following these strict rules:

1. Topic Names (Dictionary Keys):
   - Remove any leading numbers and dots (e.g., "01. Topic" → "Topic")
   - Use proper spacing between words
   - Start with a capital letter
   - No special characters or punctuation
   - Maximum 50 characters
   - Only letters, numbers, and spaces allowed

2. JSON Structure:
{
    "Topic Name": [ "subtopic 1", "subtopic 2", ... ],
    "Another Topic": [ "subtopic 1", "subtopic 2", ... ],
    ...
}

CORRECT Examples:
✅ "Financial Markets": [ ... ]
✅ "Machine Learning Fundamentals": [ ... ]
✅ "Statistical Analysis": [ ... ]

INCORRECT Examples (DO NOT DO THIS):
❌ "01. Financial_Markets": [ ... ]      (has number and underscore)
❌ "machine learning": [ ... ]           (not capitalized)
❌ "Statistical-Analysis": [ ... ]       (has hyphen)
❌ "2. Data Science": [ ... ]            (has leading number)

Return only the JSON object with properly formatted topic names as keys.""")
    
    chat = topics_model.start_chat()
    response = chat.send_message(topics_content)
    return json.loads(response.text)

def create_filename_model():
    """Create a model specifically for generating filenames."""
    filename_config = {
        "temperature": 0.7,  # Reduced for more consistent naming
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,  # Reduced since we only need short responses
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=filename_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""Generate a concise, descriptive filename for the given topic. 

REQUIREMENTS:
- Maximum 50 characters
- Use only letters, numbers, and spaces
- Start with a capital letter
- Be descriptive but concise
- Return ONLY the filename, nothing else
- ALWAYS include spaces between words
- NO special characters or punctuation
- NO file extensions

CORRECT Examples:
Input: "Hierarchical Thread Structure: Concepts of grids, blocks, and threads within CUDA"
Output: "Hierarchical Thread Structure"

Input: "Understanding the Bias-Variance Tradeoff in Machine Learning Models"
Output: "Bias Variance Tradeoff"

Input: "Ridge Regression and L2 Regularization: Mathematical Foundations & Implementation"
Output: "Ridge Regression and L2 Regularization"

Input: "Deep Analysis of Gradient Descent: Convergence Properties & Optimization"
Output: "Gradient Descent Analysis"

INCORRECT Examples (DO NOT DO THIS):
❌ "HierarchicalThreadStructure"    (Missing spaces)
❌ "Hierarchical_Thread_Structure"  (Contains underscores)
❌ "hierarchical thread structure"  (Not capitalized)
❌ "Hierarchical-Thread-Structure"  (Contains hyphens)
❌ "Hierarchical Thread Structure.md"  (Contains extension)
❌ "The Complete and Comprehensive Guide to Understanding Hierarchical Thread Structure in Modern CUDA Programming"  (Too long)

IMPORTANT:
1. Focus on SPACES between words
2. Keep it concise but meaningful
3. Return ONLY the filename
4. NO explanations or additional text
5. NO punctuation marks
6. ALWAYS start with a capital letter
7. ALWAYS use proper spacing between words""")

def save_topic_file(section_dir: Path, topic: str, content: str) -> str:
    """Save topic content to a file with proper numbering."""
    # Get suggested name from LLM
    filename_model = create_filename_model()
    chat = filename_model.start_chat()
    suggested_name = chat.send_message(topic).text.strip()
        
    # Get list of existing markdown files and their numbers
    existing_files = [f for f in section_dir.glob("*.md")]
    existing_numbers = {
        int(f.name.split('.')[0]) 
        for f in existing_files 
        if f.name[0].isdigit()
    }
    
    # Find the first available number
    next_num = 1
    while next_num in existing_numbers:
        next_num += 1
    
    topic_filename = f"{next_num:02d}. {suggested_name}.md"
    topic_path = section_dir / topic_filename
    
    # Check if a similar file already exists (ignoring numbers)
    for existing_file in existing_files:
        existing_name = ' '.join(existing_file.name.split(' ')[1:]).replace('.md', '')
        if existing_name.lower() == suggested_name.lower():
            print(f"⚠️ Similar topic already exists: {existing_file.name}")
            return existing_file.name
    
    # Save the new file
    topic_path.write_text(content, encoding='utf-8')
    return topic_filename

def create_section_directory(base_dir, section_name):
    """Create a section directory with proper sequential numbering."""
    # Remove any existing number prefixes from the input section name
    section_name = re.sub(r'^\d+\.\s*', '', section_name)
    clean_section_name = section_name.lower().strip()
    
    # Get existing section directories
    existing_dirs = [d for d in base_dir.iterdir() 
                    if d.is_dir()]
    
    # First check for existing similar sections
    for existing_dir in existing_dirs:
        # Remove the number prefix and clean up for comparison
        existing_name = re.sub(r'^\d+\.\s*', '', existing_dir.name).lower().strip()
        if existing_name == clean_section_name:
            print(f"✓ Using existing section: {existing_dir.name}")
            return existing_dir
    
    # If no existing section found, create new one with next available number
    existing_numbers = {
        int(d.name.split('.')[0]) 
        for d in existing_dirs 
        if d.name[0].isdigit()
    }
    
    next_num = 1
    while next_num in existing_numbers:
        next_num += 1
    
    section_dir_name = f"{next_num:02d}. {section_name}"
    section_dir = base_dir / section_dir_name
    
    if not section_dir.exists():
        section_dir.mkdir(parents=True)
        print(f"✓ Created new section: {section_dir_name}")
        
    return section_dir

def read_prompt_file(base_dir: Path) -> str:
    """Read the system prompt file."""
    prompt_file = base_dir / "02. Linear Classification" / "prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError("Prompt file not found at: {}".format(prompt_file))
    return prompt_file.read_text(encoding='utf-8')


def create_draft_model(prompt_file):
    """Create and configure the model for initial draft generation."""
    print(f"\nCreating draft model with prompt from: {prompt_file}")
    
    draft_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    with open(prompt_file, 'r') as system_prompt:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=draft_config,
            safety_settings=SAFETY_SETTINGS,
            system_instruction="\n".join(system_prompt.readlines()),
        )
    print("✓ Draft model created successfully")
    return model

@retry_on_error()
def generate_initial_draft(model, files, topic):
    """Generate initial draft for a topic using the provided files."""
    print(f"\nGenerating initial draft for topic: {topic[:100]}...")
    
    # Initialize chat with files
    history = []
    for file in files:
        history.append({
            "role": "user",
            "parts": [file],
        })
    
    chat = model.start_chat(history=history)
    response = chat.send_message(topic)
    print("✓ Initial draft generated")
    return response.text

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    print(f"\nUploading file: {path}")
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"✓ Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()


@retry_on_error()
def process_directory(directory: Path, llm: ChatGoogleGenerativeAI) -> None:
    """Process a single directory with the crew."""
    print(f"\nProcessing directory: {directory}")
    
    # Get PDF files
    pdf_files = get_pdf_files(directory)
    if not pdf_files:
        print("No PDF files found in the directory, skipping...")
        return

    try:
        # Read system prompt
        system_prompt = read_prompt_file(BASE_DIR)
        
        # Create draft model and upload files to Gemini
        draft_model = create_draft_model(system_prompt)
        uploaded_files = [
            upload_to_gemini(str(pdf), mime_type="application/pdf") 
            for pdf in pdf_files
        ]
        wait_for_files_active(uploaded_files)
        
        # Read topics file and parse with LLM
        topics_content = read_topics_file(directory)
        topics_dict = get_topics_dict(topics_content)
        
        # Create crew with necessary tools and configuration
        crew = StudyAgent().crew(llm=llm)
        
        # Process each section separately
        for i, (section_name, topics) in enumerate(topics_dict.items(), 1):
            numbered_section_name = f"{i:02d}. {section_name}"
            section_dir = create_section_directory(directory, numbered_section_name)
            
            # Get existing topic numbers in this section
            existing_files = [f for f in section_dir.iterdir() if f.name.endswith('.md')]
            existing_numbers = {
                int(f.name.split('.')[0]) 
                for f in existing_files 
                if f.name[0].isdigit()
            }
            
            # Process each topic with proper numbering
            for j, topic in enumerate(topics, 1):
                # Find next available topic number
                topic_num = j
                while topic_num in existing_numbers:
                    topic_num += 1
                
                # Generate initial draft
                initial_draft = generate_initial_draft(draft_model, uploaded_files, topic)
                
                # Process the draft with crew
                result = crew.kickoff(
                    inputs={
                        'directory': str(directory),
                        'section_dir': str(section_dir),
                        'initial_draft': initial_draft,
                        'topic': topic,
                        'topic_number': topic_num,
                        'section_name': numbered_section_name,
                        'topics_content': topics_content,
                        'system_prompt': system_prompt
                    }
                )
                existing_numbers.add(topic_num)
            
        print(f"✓ Successfully processed directory: {directory}")
        return result
        
    except Exception as e:
        error_msg = f"""
Time: {datetime.now()}
Directory: {directory}
Error: {str(e)}
Stack Trace:
{traceback.format_exc()}
----------------------------------------
"""
        error_file = BASE_DIR / "errors.txt"
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(error_msg)
        print(f"❌ Failed to process directory: {directory}")
        print(f"Error: {str(e)}")
        print(error_msg)

def run():
    """Run the crew following the reference script workflow."""
    # Set up Gemini LLM
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    gemini_llm = LLM(
        provider="google",
        model="gemini/gemini-2.0-flash-exp",
        temperature=0.7,
        api_key=os.environ["GOOGLE_API_KEY"],
        seed=42
    )
    
    # Get input directories
    input_directories = get_input_directories(BASE_DIR)
    if not input_directories:
        print("No valid input directories found!")
        return
    
    # Process each directory (no need to convert to Path here anymore)
    for directory in input_directories:
        process_directory(directory, gemini_llm)

if __name__ == "__main__":
    run()
