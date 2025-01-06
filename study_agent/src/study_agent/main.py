#!/usr/bin/env python
import os
import sys
import warnings
from datetime import datetime
import traceback
from pathlib import Path
from typing import List, Dict
import time
from functools import wraps
import re
import json

from study_agent.crew import StudyAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import DirectoryReadTool, FileReadTool

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

BASE_DIR = Path(__file__).parent.parent.parent

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
                        error_file.write_text(error_msg, encoding='utf-8', mode='a')
                        print(f"❌ Failed after {max_retries} attempts: {func.__name__}")
                        raise
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            return wrapper
    return decorator

def get_input_directories(base_dir: Path) -> List[Path]:
    """Get all valid input directories from the base directory."""
    input_dirs = []
    for item in base_dir.iterdir():
        if not item.is_dir() or item.name.startswith("00."):
            continue
            
        # Skip if no topics.md found
        if not any(f.name.lower() == "topics.md" for f in item.iterdir()):
            continue
            
        input_dirs.append(item)
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

def get_topics_dict(topics_content: str, llm) -> Dict[str, List[str]]:
    """Parse topics content into a dictionary of sections and topics using LLM."""
    
    # Configure LLM for filename generation
    filename_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,  # Lower temperature for more consistent naming
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024,  # Reduced since we only need short responses
        convert_system_message_to_human=True
    )

    system_prompt = """Generate a concise, descriptive filename for the given topic. 

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
7. ALWAYS use proper spacing between words"""

    # Get filename suggestion from LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": topics_content}
    ]
    
    try:
        response = filename_llm.invoke(messages)
        suggested_name = response.content.strip()
        
        # Basic validation of the suggested name
        if len(suggested_name) > 50 or not suggested_name[0].isupper():
            # Fall back to basic cleaning if validation fails
            suggested_name = ' '.join(word.capitalize() for word in topics_content.split()[:5])
            suggested_name = re.sub(r'[^\w\s]', '', suggested_name)[:50].strip()
            
        return suggested_name
        
    except Exception as e:
        print(f"❌ Failed to generate filename with LLM: {str(e)}")
        # Fall back to basic cleaning
        return ' '.join(word.capitalize() for word in topics_content.split()[:5])[:50]

def save_topic_file(section_dir: Path, topic: str, content: str) -> str:
    """Save topic content to a file with proper numbering."""
    # Get suggested name from LLM
    suggested_name = get_topics_dict(topic, None)  # Pass None since we don't need the full topics dict functionality
    
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
    prompt_file = base_dir / "00. prompts" / "Resumo.md"
    if not prompt_file.exists():
        raise FileNotFoundError("Prompt file not found at: {}".format(prompt_file))
    return prompt_file.read_text(encoding='utf-8')

@retry_on_error()
def process_directory(directory: Path, llm) -> None:
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
        
        # Read topics file and parse with LLM
        topics_content = read_topics_file(directory)
        topics_dict = get_topics_dict(topics_content, llm)
        
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
                
                # Process the topic with crew
                result = crew.kickoff(
                    inputs={
                        'directory': str(directory),
                        'section_dir': str(section_dir),
                        'pdf_files': [str(f) for f in pdf_files],
                        'topic': topic,
                        'topic_number': topic_num,  # Pass the topic number
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
        error_file.write_text(error_msg, encoding='utf-8', mode='a')
        print(f"❌ Failed to process directory: {directory}")
        print(f"Error: {str(e)}")

def run():
    """Run the crew following the reference script workflow."""
    # Set up Gemini LLM
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Get input directories
    input_directories = get_input_directories(BASE_DIR)
    if not input_directories:
        print("No valid input directories found!")
        return
    
    # Process each directory
    for directory in input_directories:
        process_directory(directory, gemini_llm)

if __name__ == "__main__":
    run()
