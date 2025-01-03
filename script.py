import os
import time
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "00. prompts", "Resumo.md")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

TARGET_FOLDERS = [
    "01. Finacial Markets"
]

# Model configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Safety settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
}

def init_gemini():
    """Initialize Gemini API with environment credentials."""
    genai.configure(api_key=GEMINI_API_KEY)

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

def create_model(prompt_file):
    """Create and configure the Gemini model."""
    print(f"\nCreating model with prompt from: {prompt_file}")
    with open(prompt_file, 'r') as system_prompt:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            system_instruction="\n".join(system_prompt.readlines()),
        )
    print("✓ Model created successfully")
    return model

def initialize_chat_session(model, files):
    """Initialize a chat session with multiple files."""
    print("\nInitializing chat session...")
    # Create initial history with all files
    history = []
    for file in files:
        history.append({
            "role": "user",
            "parts": [file],
        })
    
    chat_session = model.start_chat(history=history)
    print("✓ Chat session initialized")
    return chat_session

def process_topics(chat_session, topics):
    """Process each topic and collect responses."""
    sections = []
    total_topics = len(topics)
    print(f"\nProcessing {total_topics} topics:")
    
    for i, topic in enumerate(topics, 1):
        print(f"\nTopic {i}/{total_topics}:")
        print(f"- {topic[:100]}...")  # Print first 100 chars of topic
        response = chat_session.send_message(topic)
        sections.append(response.text)
        print("✓ Response received")
    
    print("\n✓ All topics processed successfully")
    return sections

def save_output(content, output_file):
    """Save the generated content to a file."""
    print(f"\nSaving output to: {output_file}")
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(content)
    print("✓ Output saved successfully")

def get_pdf_files_in_dir(directory):
    """Get all PDF files in the specified directory."""
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

def upload_multiple_files(file_paths):
    """Upload multiple files to Gemini."""
    uploaded_files = []
    for path in file_paths:
        uploaded_files.append(upload_to_gemini(path, mime_type="application/pdf"))
    return uploaded_files

def read_topics_file(directory):
    """Read and find the topics.md file in the given directory."""
    for file in os.listdir(directory):
        if file.lower() == 'topics.md':
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    raise FileNotFoundError("topics.md not found in the specified directory")

def get_topics_dict(topics_content):
    """Convert topics content to dictionary using Gemini."""
    # Configure model for JSON parsing
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
        system_instruction="""given the topics, return it in a json format with the following template:
        {
        "topic_name": [ "topic 1", "topic 2", ... ],
        "topic_name": [ "topic 1", "topic 2", ... ],
        ...
        }"""
    )
    
    # Get JSON response
    chat = topics_model.start_chat()
    response = chat.send_message(topics_content)
    return json.loads(response.text)

def create_diagram_model():
    """Create a model specifically for generating Mermaid diagrams."""
    diagram_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=diagram_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""You are a technical documentation expert specializing in Mermaid diagrams. Your task is to:

1. Analyze the given text
2. Identify concepts that would benefit from visual representation
3. Add Mermaid diagram code blocks where appropriate, using this format:
   ```mermaid
   [diagram code here]
   ```

Guidelines for diagrams:
- Focus on architecture and system relationships
- Use clear, descriptive labels in quotes
- Prefer flowcharts and sequence diagrams over mindmaps
- Keep diagrams focused and not too complex
- Always use proper Mermaid syntax
- Add diagrams inline where they best support the text
- Do not forget the first diagram of the text and to replace by a diagram others <image: ...> blocks

Do not modify the original text - only add Mermaid diagram blocks where helpful."""
    )

def create_section_directory(base_dir, section_name):
    """Create and return the path to a section directory."""
    section_dir = os.path.join(base_dir, section_name)
    os.makedirs(section_dir, exist_ok=True)
    return section_dir

def generate_topic_content(chat_session, topic):
    """Generate initial content for a topic."""
    return chat_session.send_message(topic)

def add_diagrams_to_content(diagram_model, content):
    """Process content to add Mermaid diagrams."""
    diagram_chat = diagram_model.start_chat()
    return diagram_chat.send_message(
        f"""Please enhance this text by adding appropriate Mermaid diagrams:

{content.text}

Remember to:
1. Keep the original text unchanged
2. Add Mermaid diagram code blocks where they would help explain concepts
3. Place diagrams in logical positions within the text
"""
    )

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
        system_instruction="""Generate a concise, descriptive filename for the given topic. Requirements:
- Maximum 50 characters
- Use only letters, numbers, and spaces
- Start with a capital letter
- Be descriptive but concise
- Return ONLY the filename, nothing else
- Don't return the filename without space between words

Example input:
"Hierarchical Thread Structure: Concepts of grids, blocks, and threads within CUDA"

Example output:
"Hierarchical Thread Structure" 

Focus on space between words!
""")

def save_topic_file(section_dir, topic, index, content):
    """Save topic content to a file."""
    # Create filename model and get suggested name
    filename_model = create_filename_model()
    chat = filename_model.start_chat()
    suggested_name = chat.send_message(topic).text.strip()
    
    # Create filename with index and suggested name
    topic_filename = f"{index:02d}. {suggested_name}.md"
    topic_path = os.path.join(section_dir, topic_filename)
    
    with open(topic_path, "w", encoding='utf-8') as f:
        f.write(content.text)
    return topic_filename

def create_math_format_model():
    """Create a model specifically for formatting mathematical notation."""
    math_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=math_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""Format all mathematical expressions using LaTeX notation within $ or $$ delimiters. 
        
Examples of replacements:
- F(X) = σ({Xk : k = 0, 1, ..., T}) → $F(X) = \sigma(\{X_k : k = 0, 1, \ldots, T\})$
- {X(t1) ∈ B1, X(t2) ∈ B2, ..., X(tk) ∈ Bk} → $\{X(t_1) \in B_1, X(t_2) \in B_2, \ldots, X(t_k) \in B_k\}$
- P(X ≤ x) → $P(X \leq x)$
- E[X] = ∫ x dF(x) → $E[X] = \int x \, dF(x)$
- σ² = E[(X - μ)²] → $\sigma^2 = E[(X - \mu)^2]$
- ∑(xi - x̄)² → $\sum(x_i - \bar{x})^2$

Guidelines:
1. Preserve all original text content
2. Only modify mathematical expressions
3. Use $ for inline math and $$ for display math
4. Format special characters: ∈ → \in, ∑ → \sum, ∫ → \int, etc.
5. Add proper subscripts: x1 → x_1, xn → x_n
6. Format Greek letters: σ → \sigma, μ → \mu
7. Use \ldots for ellipsis in math mode
8. Add proper spacing with \, where needed
9. Don't modify existing correctly formatted LaTeX expressions""")

def format_math_notation(math_model, content):
    """Process content to format mathematical notation using LaTeX."""
    math_chat = math_model.start_chat()
    return math_chat.send_message(
        f"""Please format all mathematical expressions in this text using LaTeX notation:

{content.text}

Remember to preserve all original content and only modify mathematical expressions not formatted yet."""
    )

def create_cleanup_model():
    """Create a model specifically for cleaning up prompt artifacts."""
    cleanup_config = {
        "temperature": 0.3,  # Lower temperature for more consistent cleanup
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=cleanup_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""Remove any prompt artifacts or instructions that were accidentally included in the generated text while preserving all actual content. 

Examples of text to remove:
- "Seu capítulo deve ser construído..."
- "Baseie seu capítulo exclusivamente..."
- "Organize o conteúdo logicamente..."
- "Use $ para expressões matemáticas..."
- "Adicione lemmas e corolários..."
- "Lembre-se de usar $ em vez de..."
- "Tenha cuidado para não se desviar..."
- "Exemplos técnicos devem ser apenas em Python..."
- "Não traduza nomes técnicos..."
- "Incorpore diagramas e mapas mentais..."
- "Continue explicando em detalhe..."
- "Deseja que eu continue com as próximas seções?"
- References to "o contexto" or "conforme indicado no contexto"
- Meta-instructions about formatting or structure
- Any text discussing how to use Mermaid, LaTeX, or image tags
- Instructions about how to write proofs or theorems
- References to [^X], [^Y], [^Z] without actual content
- Placeholder text like "{{ X }}" or "<X>"

Guidelines:
1. Preserve all actual content, including:
   - Mathematical formulas and equations
   - Technical explanations and proofs
   - Theorems, lemmas, and corolários
   - Diagrams and image descriptions
   - References with actual content [^n]
   - Section titles and headers
   - Mermaid diagrams with actual content
2. Remove only meta-instructions and prompt artifacts
3. Ensure the flow remains natural after removing artifacts
4. Keep mathematical notation intact ($, $$)
5. Preserve formatting (**, *, >, emojis) when used for actual content
6. Keep proof endings ($\blacksquare$) but remove instructions about using them
7. Remove any "Nota:" or "Importante:" sections that contain only instructions
8. Keep actual notes marked with emojis (⚠️❗✔️💡) if they contain technical content""")

def cleanup_prompt_artifacts(cleanup_model, content):
    """Process content to remove prompt artifacts."""
    cleanup_chat = cleanup_model.start_chat()
    return cleanup_chat.send_message(
        f"""Please clean up any prompt artifacts or instructions from this text while preserving all actual content:

{content.text}

Remember to keep all mathematical formulas, technical content, and structural elements intact."""
    )

def create_numerical_examples_model():
    """Create a model specifically for adding numerical examples."""
    examples_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=examples_config,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="""Add practical numerical examples to theoretical sections while preserving all existing content. 

Guidelines for adding examples:
1. Identify sections that would benefit from practical examples
2. Add examples after theoretical explanations using this format:

> 💡 **Exemplo Numérico:**
[Example content with actual numbers, calculations, and visualizations]

Example types to add based on linear regression topics:
- Bias-variance tradeoff calculations with specific datasets
- Ridge and Lasso regularization with different λ values
- Matrix calculations for least squares estimation
- Orthogonalization examples using Gram-Schmidt
- Principal Component Analysis (PCA) with actual data
- Cross-validation error calculations
- Parameter estimation and confidence intervals
- F-statistics and hypothesis testing examples
- Subset selection comparisons with real predictors
- Path algorithms with concrete coefficient values

Required components:
1. Use Python code with numpy/scipy/sklearn/pytorch when appropriate:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
# Example code
```

2. Include visualizations using Mermaid when helpful:
```mermaid
# Diagram showing relationships
```

3. Show mathematical calculations step by step:
$\text{Step 1: } \beta = (X^TX)^{-1}X^Ty$
$\text{Step 2: } ...$

4. Use tables for comparing methods:
| Method | MSE | R² | Parameters |
|--------|-----|----| ---------- |
| OLS    | ... | ...| ...        |
| Ridge  | ... | ...| ...        |

5. Include real-world interpretations of results

Requirements:
1. Preserve all existing content
2. Format all mathematical expressions using LaTeX
3. Use realistic parameter values
4. Show intermediate calculation steps
5. Explain the intuition behind the numbers
6. Connect examples to theoretical concepts
7. Include residual analysis where appropriate
8. Compare different methods when relevant
9. Use clear variable naming conventions
10. Add error analysis and statistical tests""")

def add_numerical_examples(examples_model, content):
    """Process content to add numerical examples where appropriate."""
    examples_chat = examples_model.start_chat()
    return examples_chat.send_message(
        f"""Please add practical numerical examples to this text where appropriate:

{content.text}

Remember to:
1. Preserve all existing content
2. Add examples only where they enhance understanding
3. Use the specified format with 💡
4. Keep all mathematical notation and references intact""")

def process_topic_section(chat_session, topics, section_name, input_dir):
    """Process topics for a specific section and save individual topic files."""
    total_topics = len(topics)
    print(f"\nProcessing {total_topics} topics for section: {section_name}")
    
    # Create models and directory
    cleanup_model = create_cleanup_model()
    examples_model = create_numerical_examples_model()
    diagram_model = create_diagram_model()
    math_model = create_math_format_model()
    section_dir = create_section_directory(input_dir, section_name)
    
    for i, topic in enumerate(topics, 1):
        print(f"\nTopic {i}/{total_topics}:")
        print(f"- {topic[:100]}...")
        
        # Generate and enhance content
        initial_response = generate_topic_content(chat_session, topic)
        print("✓ Cleaning up prompt artifacts...")
        cleaned_response = cleanup_prompt_artifacts(cleanup_model, initial_response)
        print("✓ Adding numerical examples...")
        with_examples = add_numerical_examples(examples_model, cleaned_response)
        print("✓ Adding diagrams...")
        with_diagrams = add_diagrams_to_content(diagram_model, with_examples)
        print("✓ Formatting mathematical notation...")
        final_response = format_math_notation(math_model, with_diagrams)
        
        # Save result
        filename = save_topic_file(section_dir, topic, i, final_response)
        print(f"✓ Saved topic with cleanup, examples, diagrams, and formatted math to: {filename}")

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
        input_dirs.append(dir_path)
    return input_dirs

def main():
    """Main execution flow."""
    # Initialize Gemini
    init_gemini()

    # Get all valid input directories
    input_directories = get_input_directories(BASE_DIR)
    
    if not input_directories:
        print("No valid input directories found!")
        return

    for input_dir in input_directories:
        print(f"\nProcessing directory: {input_dir}")
        
        # Get PDF files for current directory
        pdf_files = get_pdf_files_in_dir(input_dir)
        
        if not pdf_files:
            print("No PDF files found in the directory, skipping...")
            continue

        # Upload and process files
        files = upload_multiple_files(pdf_files)
        wait_for_files_active(files)

        # Create model and chat session
        model = create_model(PROMPT_FILE)
        chat_session = initialize_chat_session(model, files)

        # Get topics dictionary
        try:
            topics_content = read_topics_file(input_dir)
            topics_dict = get_topics_dict(topics_content)

            # Process each section separately
            for i, (section_name, topics) in enumerate(topics_dict.items(), 1):
                # Add section number to section name
                numbered_section_name = f"{i:02d}. {section_name}"
                # Process topics for this section
                process_topic_section(chat_session, topics, numbered_section_name, input_dir)
                
        except FileNotFoundError as e:
            print(f"Error processing directory {input_dir}: {e}")
            continue

if __name__ == "__main__":
    main()