import os
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from functools import wraps

# Import missing functions
from script import (
    init_gemini,
    upload_to_gemini,
    wait_for_files_active,
    create_model,
    initialize_chat_session,
    get_topics_dict,
    create_section_directory,
    process_topic_section,
    get_input_directories,
    retry_on_error
)

# Import missing constants
from script import (
    BASE_DIR,
    PROMPT_FILE,
    SAFETY_SETTINGS,
    GENERATION_CONFIG
)

# Update TARGET_FOLDERS constant if needed
TARGET_FOLDERS = []  # Empty to process all folders

def get_pdf_files_in_dir(directory):
    """Get PDF files in the specified directory that start with 'Det -'."""
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf') and "Det -" in file:
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

def create_intersection_model():
    """Create a model specifically for finding topic intersections."""
    intersection_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=intersection_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""Analyze two lists of topics and find meaningful intersections between them.

Your task is to:
1. Identify topics that are semantically similar or related in its theory
2. Create a new list of intersection topics that captures shared concepts
3. Return the result as a JSON object with proper topic organization

Guidelines:
1. Look for conceptual similarities even if wording differs
2. Consider hierarchical relationships between topics
3. Merge related subtopics when appropriate
4. Maintain academic rigor and technical precision
5. Follow the same topic formatting rules as before

Output Format:
{
    "Topic Name": ["subtopic 1", "subtopic 2", ...],
    ...
}

Example:
List 1:
- "Financial Markets and Instruments"
- "Risk Management in Trading"

List 2:
- "Market Structure Analysis"
- "Trading Risk Assessment"

Should produce:
{
    "Financial Market Analysis": [
        "Market structure and instruments",
        "Risk assessment in trading operations"
    ]
}

Remember:
- Topic names should be properly formatted (capitalized, no numbers/special chars)
- Subtopics should be specific and well-defined
- Focus on meaningful intersections, not just keyword matches
- Consider the broader context of each topic""")

def extract_topics_from_pdf(model, pdf_file):
    """Extract topics from a PDF file using the topics prompt."""
    print(f"📄 Extracting topics from {os.path.basename(pdf_file)}...")
    
    # Upload PDF and create chat session
    pdf = upload_to_gemini(pdf_file, mime_type="application/pdf")
    wait_for_files_active([pdf])
    print("✓ PDF uploaded successfully")
    
    # Read topics prompt
    with open(PROMPT_FILE, 'r') as f:
        topics_prompt = f.read()
    
    chat = model.start_chat(history=[{"role": "user", "parts": [pdf]}])
    response = chat.send_message(topics_prompt)
    
    # Convert response to topics dictionary
    topics_dict = get_topics_dict(response.text)
    print(f"✓ Extracted {len(topics_dict)} topics")
    return topics_dict

def find_topic_intersections(intersection_model, topics1, topics2):
    """Find intersections between two sets of topics."""
    print(f"🔄 Finding intersections between {len(topics1)} and {len(topics2)} topics...")
    
    chat = intersection_model.start_chat()
    response = chat.send_message(f"""Find meaningful intersections between these two topic lists:

List 1:
{json.dumps(topics1, indent=2)}

List 2:
{json.dumps(topics2, indent=2)}

Return only the JSON object with intersection topics.""")
    
    intersection_topics = json.loads(response.text)
    print(f"✓ Found {len(intersection_topics)} intersection topics")
    return intersection_topics

def find_best_subtopic_folder(model, topic, section_dir):
    """Find the most appropriate subtopic folder based on content analysis."""
    chat = model.start_chat()
    response = chat.send_message(f"""Analyze this topic and determine the most appropriate subtopic folder from the available options.

Topic: {topic}

Available folders:
{[d for d in os.listdir(section_dir) if os.path.isdir(os.path.join(section_dir, d))]}

Return only the folder name that best matches the topic content.""")
    
    return response.text.strip()

def main():
    """Main execution flow."""
    print("\n🚀 Starting intersection analysis...\n")
    
    # Initialize Gemini
    init_gemini()
    print("✓ Gemini API initialized")
    
    # Create specialized models
    intersection_model = create_intersection_model()
    base_model = create_model(PROMPT_FILE)
    print("✓ Models created successfully")
    
    # Get all valid input directories
    input_directories = get_input_directories(BASE_DIR)
    
    if not input_directories:
        print("No valid input directories found!")
        return
    
    for input_dir in input_directories:
        print(f"\n📁 Processing directory: {input_dir}")
        
        pdf_files = get_pdf_files_in_dir(input_dir)
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("⚠️ No PDF files found in the directory, skipping...")
            continue
        
        for pdf_file in pdf_files:
            print(f"\n📎 Processing PDF: {os.path.basename(pdf_file)}")
            
            # Extract topics from PDF
            pdf_topics = extract_topics_from_pdf(base_model, pdf_file)
            
            # Get or extract existing topics
            topics_file = os.path.join(input_dir, "topics.md")
            if os.path.exists(topics_file):
                with open(topics_file, 'r') as f:
                    existing_topics = get_topics_dict(f.read())
            else:
                # Extract topics from other PDFs in directory
                other_pdfs = [f for f in get_pdf_files_in_dir(input_dir) if f != pdf_file]
                existing_topics = {}
                for other_pdf in other_pdfs:
                    other_topics = extract_topics_from_pdf(base_model, other_pdf)
                    existing_topics.update(other_topics)
            
            # Find intersections
            intersection_topics = find_topic_intersections(
                intersection_model, pdf_topics, existing_topics)
            
            print(f"\n🔍 Processing {len(intersection_topics)} intersection topics...")
            for section_name, topics in intersection_topics.items():
                print(f"\n📑 Processing section: {section_name}")
                section_dir = create_section_directory(input_dir, section_name)
                
                # Create chat session for content generation
                files = [upload_to_gemini(pdf_file, mime_type="application/pdf")]
                wait_for_files_active(files)
                chat_session = initialize_chat_session(base_model, files)
                
                # Process each topic
                process_topic_section(chat_session, topics, section_name, section_dir)
                print(f"✓ Completed processing {len(topics)} topics in section")
    
    print("\n✨ Intersection analysis completed successfully!")

if __name__ == "__main__":
    main() 