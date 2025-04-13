import os
import re
from dotenv import load_dotenv

load_dotenv()

from pollo.agents.draft.writer import generate_drafts_from_topics

# Define the base directory where topic folders are located
base_dir = "."  # Change this to your actual base directory path

# Define directories to exclude from processing
EXCLUDE_DIRS = []

# Define target directories to process (if empty, process all valid directories)
TARGET_DIRS = [
]

# Define target directory numbers to process (if empty, this filter is not applied)
# Example: ["01", "03", "07"] to process folders starting with these numbers
TARGET_NUMBERS = [
    "17", "18", "19", "20", "21", "22", "23", "24"
]

# Get all directories in the base directory
all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Filter directories matching the pattern "[0-9+]. [Topic Name]"
topic_dirs = [d for d in all_dirs if re.match(r"^\d+\.\s+.+$", d)]

# Apply filtering logic:
# 1. If TARGET_DIRS is not empty, only process those specific directories
# 2. Else if TARGET_NUMBERS is not empty, only process directories with matching numbers
# 3. Otherwise, exclude directories that are in the EXCLUDE_DIRS list
if TARGET_DIRS:
    topic_dirs = [d for d in topic_dirs if d in TARGET_DIRS]
elif TARGET_NUMBERS:
    # Extract the number prefix from directory names and check against TARGET_NUMBERS
    topic_dirs = [d for d in topic_dirs if re.match(r"^(\d+)\.", d).group(1) in TARGET_NUMBERS]
else:
    topic_dirs = [d for d in topic_dirs if d not in EXCLUDE_DIRS]

# Define the perspectives for all topics
basic_perspective = "Focus on the foundational concepts of machine learning, including basic terminology, intuitive understanding, and real-world applications. Explore how these algorithms work at a high level, their advantages, limitations, and when to use them in practical scenarios."
advanced_perspective = "Focus on the mathematical foundations and statistical theory behind machine learning methods, including probability theory, optimization techniques, and algorithm complexity. Explore model assumptions, convergence properties, and theoretical guarantees of different approaches."

# Process each topic directory
for directory in topic_dirs:
    print(f"Processing directory: {directory}")
    generate_drafts_from_topics(
        directory=directory,
        perspectives=[
            basic_perspective,
            advanced_perspective
        ],
        json_per_perspective=2,
        branching_factor=1
    )
    
print(f"Processed {len(topic_dirs)} topic directories")