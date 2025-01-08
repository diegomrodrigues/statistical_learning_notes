import yaml
import os
from pathlib import Path
from agent.processor import TaskProcessor
from agent.chain import TaskChain, ChainStep
import json
from agent.filename_handler import FilenameHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import re

def get_pdf_files(directory: Path) -> list[Path]:
    """Get all PDF files in the directory."""
    return list(directory.glob("*.pdf"))

def read_topics_file(directory: Path) -> str:
    """Read and find the topics.md file in the given directory."""
    topics_file = directory / "topics.md"
    if topics_file.exists():
        return topics_file.read_text(encoding='utf-8')
    raise FileNotFoundError("topics.md not found in the specified directory")

def process_topic_wrapper(args) -> tuple[str, bool]:
    """Wrapper function for process_topic to work with ThreadPoolExecutor."""
    directory, section_name, topic, content, processor, tasks_config = args
    success = process_topic(directory, section_name, topic, content, processor, tasks_config)
    return topic, success

def process_section_topic(directory: Path, topic: str, pdf_files: list[Path],
                         processor: TaskProcessor, tasks_config: dict) -> Optional[tuple[str, str]]:
    """Process a single topic within a section."""
    print(f"Processing topic: {topic[:50]}...")
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
    
    chain = TaskChain(processor, tasks_config, steps)
    initial_content = f"X = {topic}"
    
    try:
        final_content = chain.run(initial_content)
        if final_content:
            return topic, final_content
    except Exception as e:
        print(f"❌ Error processing topic {topic}: {str(e)}")
    
    return None

def process_topic(directory: Path, section_name: str, topic: str, content: str, 
                 processor: TaskProcessor, tasks_config: dict) -> bool:
    """Process a single topic and save it to a file."""
    try:
        # Initialize filename handler
        filename_handler = FilenameHandler(processor, tasks_config)

        # Create or get section directory
        section_dir = filename_handler.create_section_directory(directory, section_name)
        
        # Generate filename within the section directory
        result = filename_handler.generate_filename(section_dir, topic)
        
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

def process_directory(directory: Path, processor: TaskProcessor, tasks_config: dict, max_workers: int = 3) -> None:
    """Process a single directory with its topics using parallel processing."""
    print(f"\nProcessing directory: {directory}")
    
    pdf_files = get_pdf_files(directory)
    if not pdf_files:
        print("No PDF files found in the directory, skipping...")
        return

    try:
        topics_content = read_topics_file(directory)
        topics_steps = [ChainStep(name="Parse Topics", tasks=["parse_topics_task"], expect_json=True)]
        topics_chain = TaskChain(processor, tasks_config, topics_steps)
        topics_result = topics_chain.run(topics_content)
        
        if not topics_result:
            raise Exception("Failed to parse topics")
            
        topics = json.loads(topics_result)
        
        # Process sections in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # First, submit all topic processing tasks
            future_to_topic = {}
            for section_name, section_topics in topics.items():
                print(f"\nSubmitting section: {section_name}")
                for topic in section_topics:
                    future = executor.submit(
                        process_section_topic,
                        directory,
                        topic,
                        pdf_files,
                        processor,
                        tasks_config
                    )
                    future_to_topic[future] = (section_name, topic)

            # Process completed topics and save results
            topic_results = []
            for future in as_completed(future_to_topic):
                section_name, topic = future_to_topic[future]
                try:
                    result = future.result()
                    if result:
                        topic_results.append((section_name, result[0], result[1]))
                    else:
                        print(f"❌ Failed to process topic: {topic}")
                except Exception as e:
                    print(f"❌ Error processing topic {topic}: {str(e)}")

            # Save results in parallel
            save_futures = []
            for section_name, topic, content in topic_results:
                save_futures.append(
                    executor.submit(
                        process_topic_wrapper,
                        (directory, section_name, topic, content, processor, tasks_config)
                    )
                )

            # Wait for all saves to complete
            for future in as_completed(save_futures):
                topic, success = future.result()
                if success:
                    print(f"✔️ Topic processed and saved with success: {topic}")
                else:
                    print(f"❌ Failed to save topic: {topic}")

    except Exception as e:
        print(f"❌ Failed to process directory: {directory}")
        print(f"Error: {str(e)}")
        raise

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
    
    # Add max_workers configuration
    max_workers = 3  # Configurable number of parallel workers
    
    # Process each target directory
    for folder in target_folders:
        directory = base_dir / folder
        if directory.exists():
            process_directory(directory, processor, tasks_config, max_workers)
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main() 