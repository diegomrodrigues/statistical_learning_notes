from pathlib import Path
import os
import yaml
from agent.processor import TaskProcessor
from agent.directory_processor import DirectoryProcessor

BASE_DIR = "/content/statistical_learning_notes"

CONTEXT = "Advanced Study of Machine Learning and Statistical Learning Theory"
PERSPECTIVES = [
    "Foque nos fundamentos matemáticos e estatísticos, incluindo teoria da probabilidade, álgebra linear, otimização e provas matemáticas relevantes.",
    "Foque nos conceitos fundamentais de machine learning e aprendizado estatístico, incluindo bias-variance tradeoff, regularização, validação cruzada e métricas de avaliação.",
    "Foque nos aspectos computacionais e de implementação, incluindo complexidade algorítmica, otimização de código e escalabilidade."
]

TARGET_FOLDERS = [
    "02. Linear Classification"
]

EXCLUDED_FOLDERS = []

# Processing parameters
NUM_TOPICS = None
MAX_WORKERS = 3
JSONS_PER_PERSPECTIVE = 2
NUM_CONSOLIDATION_STEPS = 3
MAX_PREVIOUS_TOPICS = 5

def load_tasks_config(tasks_dir: str = './agent/tasks') -> dict:
    """Load all YAML files from the tasks directory into a single config dictionary."""
    tasks_config = {}
    tasks_path = Path(tasks_dir)
    
    for yaml_file in tasks_path.glob('*.yaml'):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            tasks_config.update(config)
    
    return tasks_config

def get_numbered_folders(base_dir: Path) -> list[str]:
    """
    Get all numbered folders from the base directory, excluding specified folders.
    If TARGET_FOLDERS is specified, only return those folders.
    Returns folders sorted numerically.
    """
    if TARGET_FOLDERS:
        return sorted([folder for folder in TARGET_FOLDERS if folder not in EXCLUDED_FOLDERS])
    
    folders = [
        folder.name for folder in base_dir.iterdir()
        if folder.is_dir() 
        and folder.name.strip()[0].isdigit()
        and folder.name not in EXCLUDED_FOLDERS
    ]
    return sorted(folders)

def main():
    # Load configuration and initialize processor
    tasks_config = load_tasks_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    processor = TaskProcessor(api_key=api_key)
    directory_processor = DirectoryProcessor(processor, tasks_config, CONTEXT)
    
    # Define base directory and settings
    base_dir = Path(BASE_DIR)
    target_folders = get_numbered_folders(base_dir)
    
    # Process each target directory
    for folder in target_folders:
        directory = base_dir / folder
        if directory.exists():
            directory_processor.process_with_topics(
                directory,
                perspectives=PERSPECTIVES,
                num_topics=NUM_TOPICS or len(PERSPECTIVES),
                max_workers=MAX_WORKERS,
                jsons_per_perspective=JSONS_PER_PERSPECTIVE,
                num_consolidation_steps=NUM_CONSOLIDATION_STEPS,
                max_previous_topics=MAX_PREVIOUS_TOPICS
            )
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main()