import os
import PyPDF2
from pathlib import Path

def split_pdf_by_sections(pdf_path, sections_dict, output_folder=".", skip_pages=0):
    """
    Split a PDF into multiple PDFs based on section definitions
    
    Args:
        pdf_path (str): Path to the PDF file
        sections_dict (dict): Dictionary with section names as keys and (start_page, end_page) tuples as values
        output_folder (str): Directory where section folders will be created (default: current directory)
        skip_pages (int): Number of initial pages to skip (for TOC, etc.) (default: 0)
    """
    # Get the PDF filename without extension
    pdf_filename = Path(pdf_path).stem
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the original PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Sort sections by start page to assign numbers in order
        sorted_sections = sorted(sections_dict.items(), key=lambda x: x[1][0])
        
        for i, (section_name, (start_page, end_page)) in enumerate(sorted_sections, 1):
            start_page += 13
            end_page += 13
            
            # Create folder with format: "01. section a/"
            folder_name = f"{i:02d}. {section_name}"
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create a new PDF writer
            pdf_writer = PyPDF2.PdfWriter()
            
            # Extract pages (PyPDF2 is 0-indexed, but user input is 1-indexed)
            # Also adjust for skipped pages
            for page_num in range(start_page - 1 + skip_pages, end_page + skip_pages):
                if page_num < len(pdf_reader.pages):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                    
            # Create output file path
            output_filename = f"{pdf_filename}_{start_page}-{end_page}.pdf"
            output_path = os.path.join(folder_path, output_filename)
            
            # Write the extracted pages to a new PDF
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
                
            print(f"Created: {output_path}")

if __name__ == "__main__":
    output_folder = "."

    sections = {
        "Generative models for discrete data": (65, 96),
        "Gaussian models": (97, 148),
        "Bayesian statistics": (149, 190),
        "Frequentist statistics": (191, 216),
        "Linear regression": (217, 244),
        "Logistic regression": (245, 280),
        "Generalized linear models and the exponential family": (281, 306),
        "Directed graphical models (Bayes nets)": (307, 336),
        "Mixture models and the EM algorithm": (337, 380),
        "Latent linear models": (381, 420),
        "Sparse linear models": (421, 478),
        "Kernels": (479, 514),
        "Gaussian processes": (515, 542),
        "Adaptive basis function models": (543, 588),
        "Markov and hidden Markov models": (589, 630),
        "State space models": (631, 660),
        "Undirected graphical models (Markov random fields)": (661, 706),
        "Exact inference for graphical models": (707, 730),
        "Variational inference": (731, 766),
        "More variational inference": (767, 804),
        "Markov chain Monte Carlo (MCMC) inference": (837, 874),
        "Clustering": (875, 906),
        "Graphical model structure learning": (907, 944),
        "Latent variable models for discrete data": (945, 991)
    }

    # Replace with your PDF path
    pdf_path = "ML Machine Learning-A Probabilistic Perspective.pdf"
    
    # Example with output folder and skipping 5 pages of TOC
    split_pdf_by_sections(
        pdf_path, 
        sections, 
        output_folder=output_folder, 
        skip_pages=18
    )