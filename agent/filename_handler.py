from pathlib import Path
from typing import Set
from dataclasses import dataclass
from .chain import TaskChain, ChainStep
from .processor import TaskProcessor

@dataclass
class FilenameResult:
    filename: str
    path: Path
    exists: bool = False

class FilenameHandler:
    def __init__(self, processor: TaskProcessor, tasks_config: dict):
        self.processor = processor
        self.tasks_config = tasks_config

    def _get_existing_numbers(self, directory: Path) -> Set[int]:
        """Get set of existing file numbers in the directory."""
        existing_files = [f for f in directory.glob("*.md")]
        return {
            int(f.name.split('.')[0]) 
            for f in existing_files 
            if f.name[0].isdigit()
        }

    def _check_similar_exists(self, directory: Path, suggested_name: str) -> str:
        """Check if a similar filename exists and return it if found."""
        for existing_file in directory.glob("*.md"):
            existing_name = ' '.join(existing_file.name.split(' ')[1:]).replace('.md', '')
            if existing_name.lower() == suggested_name.lower():
                return existing_file.name
        return False

    def generate_filename(self, directory: Path, topic: str) -> FilenameResult:
        """Generate a filename for the topic with proper numbering."""
        # Create chain for filename generation
        steps = [
            ChainStep(
                name="Generate Filename",
                tasks=["generate_filename_task"]
            )
        ]
        chain = TaskChain(self.processor, self.tasks_config, steps)
        
        # Get suggested name
        suggested_name = chain.run(topic)
        if not suggested_name:
            raise ValueError("Failed to generate filename")
        suggested_name = suggested_name.strip()
        
        # Check for existing similar file
        existing_file = self._check_similar_exists(directory, suggested_name)
        if existing_file:
            return FilenameResult(
                filename=existing_file,
                path=directory / existing_file,
                exists=True
            )
        
        # Get next available number
        existing_numbers = self._get_existing_numbers(directory)
        next_num = 1
        while next_num in existing_numbers:
            next_num += 1
        
        # Create new filename
        new_filename = f"{next_num:02d}. {suggested_name}.md"
        return FilenameResult(
            filename=new_filename,
            path=directory / new_filename,
            exists=False
        ) 