import os
import time
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from functools import wraps
import traceback
from datetime import datetime
import re

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
                        # Log the error
                        error_msg = f"""
Time: {datetime.now()}
Function: {func.__name__}
Error: {str(e)}
Stack Trace:
{traceback.format_exc()}
----------------------------------------
"""
                        with open(os.path.join(BASE_DIR, "errors.txt"), "a", encoding='utf-8') as f:
                            f.write(error_msg)
                        print(f"❌ Failed after {max_retries} attempts: {func.__name__}")
                        raise
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
        return wrapper
    return decorator

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
    """Get PDF files in the specified directory that start with 'Ref -'."""
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf') and "Ref -" in file:
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

Return only the JSON object with properly formatted topic names as keys."""
    )
    
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

def get_next_topic_number(base_dir):
    """Get the next available topic number in the base directory."""
    max_num = 0
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)):
            try:
                num = int(item.split('.')[0])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
    return max_num + 1

def get_next_subtopic_number(topic_dir):
    """Get the next available subtopic number in the topic directory."""
    max_num = 0
    for item in os.listdir(topic_dir):
        if os.path.isdir(os.path.join(topic_dir, item)):
            try:
                num = int(item.split('.')[0])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
    return max_num + 1

def create_section_directory(base_dir, section_name):
    """Create a section directory with proper sequential numbering."""
    # Remove any existing number prefixes from the input section name
    section_name = re.sub(r'^\d+\.\s*', '', section_name)
    clean_section_name = section_name.lower().strip()
    
    # Get existing section directories
    existing_dirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
    
    # First check for existing similar sections
    for existing_dir in existing_dirs:
        # Remove the number prefix and clean up for comparison
        existing_name = re.sub(r'^\d+\.\s*', '', existing_dir).lower().strip()
        if existing_name == clean_section_name:
            print(f"✓ Using existing section: {existing_dir}")
            return os.path.join(base_dir, existing_dir)
    
    # If no existing section found, create new one with next available number
    existing_numbers = set()
    for dir_name in existing_dirs:
        try:
            num = int(dir_name.split('.')[0])
            existing_numbers.add(num)
        except (ValueError, IndexError):
            continue
    
    next_num = 1
    while next_num in existing_numbers:
        next_num += 1
    
    section_dir_name = f"{next_num:02d}. {section_name}"
    section_dir = os.path.join(base_dir, section_dir_name)
    
    if not os.path.exists(section_dir):
        os.makedirs(section_dir)
        print(f"✓ Created new section: {section_dir_name}")
        
    return section_dir

@retry_on_error()
def generate_topic_content(chat_session, topic):
    """Generate initial content for a topic."""
    return chat_session.send_message(topic)

def add_diagrams_to_content(diagram_model, content):
    """Process content to add Mermaid diagrams."""
    diagram_chat = diagram_model.start_chat()
    return diagram_chat.send_message(
        f"""Please enhance this text by adding appropriate Mermaid diagrams. Focus on creating sophisticated technical diagrams that support advanced mathematical and statistical concepts.

{content.text}

Guidelines for diagram creation:

1. Mathematical and Statistical Concepts:
   - Represent complex mathematical relationships and dependencies
   - Visualize statistical distributions and their properties
   - Illustrate theoretical frameworks and mathematical proofs
   - Show parameter spaces and optimization landscapes

2. Algorithm Visualization:
   - Detail computational flows in statistical algorithms
   - Break down complex mathematical formulas into components
   - Illustrate iterative processes in numerical methods
   - Represent matrix operations and transformations

3. Model Architecture:
   - Show hierarchical relationships in statistical models
   - Illustrate model selection processes
   - Visualize regularization paths
   - Represent cross-validation schemes

4. Theoretical Relationships:
   - Connect mathematical theorems, lemmas, and corollaries
   - Show proof structures and logical dependencies
   - Illustrate theoretical trade-offs
   - Represent abstract mathematical spaces

⚠️ CRITICAL FORMATTING REQUIREMENTS:
1. ALWAYS use double quotes (" ") around ALL text in Mermaid diagrams
2. AVOID losangles, decision nodes, and mind map structures
3. Focus on architectural and conceptual relationships
4. Break down complex formulas into their components

Example structures:

```mermaid
graph TD
    subgraph "Mathematical Decomposition"
        direction TB
        A["Complete Formula: MSE = Bias² + Variance + ε"]
        B["Bias Component: (E[f̂(x)] - f(x))²"]
        C["Variance Component: E[(f̂(x) - E[f̂(x)])²]"]
        D["Irreducible Error: var(ε)"]
        A --> B
        A --> C
        A --> D
    end
```

```mermaid
graph LR
    subgraph "Ridge Regression Components"
        direction LR
        A["Loss Function"] --> B["RSS Term: ||y - Xβ||²"]
        A --> C["Penalty Term: λ||β||²"]
        B --> D["Optimization Objective"]
        C --> D
    end
```

```mermaid
graph TB
    subgraph "Theoretical Framework"
        A["Main Theorem"] --> B["Supporting Lemma 1"]
        A --> C["Supporting Lemma 2"]
        B & C --> D["Resulting Corollary"]
        D --> E["Mathematical Implications"]
    end
```

Requirements:
1. Keep all original text content intact
2. Add diagrams only where they enhance mathematical understanding
3. Use proper mathematical notation in diagram labels
4. Place diagrams at logical breaks in the text
5. Ensure diagrams are technically precise and academically rigorous
6. Focus on theoretical and mathematical aspects over practical implementations
7. Use subgraphs to group related concepts
8. Include clear directional relationships
9. Add mathematical expressions in quotes when needed
10. Maintain consistency with LaTeX notation used in the text
11. ALWAYS use double quotes for ALL text in diagrams
12. Focus on breaking down complex concepts rather than decision flows

AVOID:
- Decision diamonds (losangles)
- Yes/No branches
- Mind map structures
- Flowchart decision points
- Simple sequential flows

PREFERRED:
- Mathematical decompositions
- Component relationships
- Theoretical hierarchies
- Formula breakdowns
- Architectural structures

Remember: Diagrams should elevate the academic rigor of the text, not simplify it."""
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

def save_topic_file(section_dir, topic, index, content):
    """Save topic content to a file with proper numbering."""
    # Create filename model and get suggested name
    filename_model = create_filename_model()
    chat = filename_model.start_chat()
    suggested_name = chat.send_message(topic).text.strip()
    
    # Get list of existing markdown files and their numbers
    existing_files = [f for f in os.listdir(section_dir) if f.endswith('.md')]
    existing_numbers = set()
    
    for file in existing_files:
        try:
            num = int(file.split('.')[0])
            existing_numbers.add(num)
        except (ValueError, IndexError):
            continue
    
    # Find the first available number
    next_num = 1
    while next_num in existing_numbers:
        next_num += 1
    
    topic_filename = f"{next_num:02d}. {suggested_name}.md"
    topic_path = os.path.join(section_dir, topic_filename)
    
    # Check if a similar file already exists (ignoring numbers)
    for existing_file in existing_files:
        existing_name = ' '.join(existing_file.split(' ')[1:]).replace('.md', '')
        if existing_name.lower() == suggested_name.lower():
            print(f"⚠️ Similar topic already exists: {existing_file}")
            return existing_file
    
    # Save the new file
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
        system_instruction=r"""Format all mathematical expressions using LaTeX notation within $ or $$ delimiters. 
        
Examples of mathematical replacements:
- F(X) = σ({Xk : k = 0, 1, ..., T}) → $F(X) = \sigma(\{X_k : k = 0, 1, \ldots, T\})$
- {X(t1) ∈ B1, X(t2) ∈ B2, ..., X(tk) ∈ Bk} → $\{X(t_1) \in B_1, X(t_2) \in B_2, \ldots, X(t_k) \in B_k\}$
- P(X ≤ x) → $P(X \leq x)$
- E[X] = ∫ x dF(x) → $E[X] = \int x \, dF(x)$
- σ² = E[(X - μ)²] → $\sigma^2 = E[(X - \mu)^2]$
- ∑(xi - x̄)² → $\sum(x_i - \bar{x})^2$

⚠️ CURRENCY FORMATTING (IMPORTANT):
Currency symbols must be escaped to prevent markdown conflicts!

CORRECT Currency Examples:
- R$ 100,00 → R\\$ 100,00
- $ 50.00 → \\$ 50.00
- The price is R$ 75,50 → The price is R\\$ 75,50
- Cost: $ 25.99 → Cost: \\$ 25.99

INCORRECT Currency Examples (DO NOT DO THIS):
❌ R$ 100,00 (unescaped R$)
❌ $ 50.00 (unescaped $)
❌ R\$ 100,00 (single backslash)
❌ \$ 50.00 (single backslash)

Guidelines:
1. Preserve all original text content
2. Only modify mathematical expressions and currency symbols
3. Use $ for inline math and $$ for display math
4. Format special characters: ∈ → \in, ∑ → \sum, ∫ → \int, etc.
5. Add proper subscripts: x1 → x_1, xn → x_n
6. Format Greek letters: σ → \sigma, μ → \mu
7. Use \ldots for ellipsis in math mode
8. Add proper spacing with \, where needed
9. Don't modify existing correctly formatted LaTeX expressions
10. ALWAYS escape currency symbols with double backslash:
    - R$ → R\\$
    - $ → \\$

Remember: Currency symbols need double backslashes to display correctly in markdown!""")

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

@retry_on_error()
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
    
    failed_topics = []
    
    for i, topic in enumerate(topics, 1):
        print(f"\nTopic {i}/{total_topics}:")
        print(f"- {topic[:100]}...")
        
        try:
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
            
        except Exception as e:
            error_msg = f"""
Time: {datetime.now()}
Section: {section_name}
Topic: {topic}
Error: {str(e)}
Stack Trace:
{traceback.format_exc()}
----------------------------------------
"""
            with open(os.path.join(BASE_DIR, "errors.txt"), "a", encoding='utf-8') as f:
                f.write(error_msg)
            failed_topics.append((topic, str(e)))
            print(f"❌ Failed to process topic: {topic[:100]}...")
            continue
    
    if failed_topics:
        print(f"\n⚠️ Failed to process {len(failed_topics)} topics in section {section_name}:")
        for topic, error in failed_topics:
            print(f"- {topic[:100]}: {error}")

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