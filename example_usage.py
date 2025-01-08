import yaml
import os
from pathlib import Path
from agent.processor import TaskProcessor
from agent.chain import TaskChain, ChainStep
import json
from agent.filename_handler import FilenameHandler

def get_pdf_files(directory: Path) -> list[Path]:
    """Get all PDF files in the directory."""
    return list(directory.glob("*.pdf"))

def read_topics_file(directory: Path) -> str:
    """Read and find the topics.md file in the given directory."""
    topics_file = directory / "topics.md"
    if topics_file.exists():
        return topics_file.read_text(encoding='utf-8')
    raise FileNotFoundError("topics.md not found in the specified directory")

def process_directory(directory: Path, processor: TaskProcessor, tasks_config: dict) -> None:
    """Process a single directory with its topics."""
    print(f"\nProcessing directory: {directory}")
    
    # Get PDF files
    pdf_files = get_pdf_files(directory)
    if not pdf_files:
        print("No PDF files found in the directory, skipping...")
        return

    try:
        # Read topics content
        topics_content = read_topics_file(directory)
        
        # First chain: Generate structured topics
        topics_steps = [
            ChainStep(
                name="Parse Topics",
                tasks=["parse_topics_task"],
                expect_json=True
            )
        ]
        
        topics_chain = TaskChain(processor, tasks_config, topics_steps)
        topics_result = topics_chain.run(topics_content)
        
        if not topics_result:
            raise Exception("Failed to parse topics")
            
        topics = json.loads(topics_result)
        
        # Second chain: Process each topic
        for section_name, section_topics in topics.items():
            print(f"\nProcessing section: {section_name}")
            
            for topic in section_topics:                    
                # Define the processing steps for this topic
                steps = [
                    ChainStep(
                        name="Generate Initial Draft",
                        tasks=["generate_draft_task"],
                        input_files=pdf_files
                    ),
                    ChainStep(
                        name="Review and Enhance",
                        tasks=[
                            "cleanup_task",
                            "generate_examples_task",
                            "create_diagrams_task",
                            "format_math_task"
                        ]
                    )
                ]
                
                # Initialize chain
                chain = TaskChain(processor, tasks_config, steps)
                
                # Run the chain for this topic
                initial_content = f"X = {topic}"
                try:
                    final_content = chain.run(initial_content)
                    if final_content:
                        success = process_topic(
                            directory=directory,
                            topic=topic,
                            content=final_content,
                            processor=processor,
                            tasks_config=tasks_config
                        )
                        if success:
                            print(f"✔️ Topic processed with success: {topic}")
                    else:
                        print(f"❌ Failed to process topic: {topic}")
                except Exception as e:
                    print(f"❌ Error processing topic {topic}: {str(e)}")
                    continue

    except Exception as e:
        print(f"❌ Failed to process directory: {directory}")
        print(f"Error: {str(e)}")
        raise

def process_topic(directory: Path, topic: str, content: str, 
                 processor: TaskProcessor, tasks_config: dict) -> bool:
    """Process a single topic and save it to a file."""
    try:
        # Initialize filename handler
        filename_handler = FilenameHandler(processor, tasks_config)
        
        # Generate filename
        result = filename_handler.generate_filename(directory, topic)
        
        if result.exists:
            print(f"⚠️ Similar topic already exists: {result.filename}")
            return True
        
        # Save content to file
        result.path.write_text(content, encoding='utf-8')
        print(f"✔️ Saved topic to: {result.filename}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to process topic: {str(e)}")
        return False

def main():
    # Load tasks configuration
    with open('agent/tasks.yaml', 'r') as f:
        tasks_config = yaml.safe_load(f)
    
    # Initialize processor
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    processor = TaskProcessor(api_key=api_key)
    
    # Define base directory and target folders
    base_dir = Path("/content/statistical_learning_notes")
    target_folders = [
        "08. Random Forests"
    ]
    
    # Process each target directory
    for folder in target_folders:
        directory = base_dir / folder
        if directory.exists():
            process_directory(directory, processor, tasks_config)
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main() 