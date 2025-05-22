import re
from typing import List, Tuple
from strands import tool
import subprocess

subprocess.run(['pip', 'install', 'PyPDF2'], capture_output=True, text=True) 
from PyPDF2 import PdfReader

@tool
def find_sentences_with_word(pdf_path: str, search_word: str) -> List[Tuple[int, str]]:
    """
    Extracts text from a PDF file and finds all sentences containing the specified search word.
    
    Args:
        pdf_path (str): The file path to the PDF document
        search_word (str): The word to search for in the document
        
    Returns:
        List[Tuple[int, str]]: A list of tuples containing:
            - Page number (1-based indexing)
            - The complete sentence containing the search word
            
    Raises:
        FileNotFoundError: If the PDF file is not found
        Exception: If there's an error processing the PDF
        
    Example:
        >>> results = find_sentences_with_word("document.pdf", "Python")
        >>> for page_num, sentence in results:
        ...     print(f"Page {page_num}: {sentence}")
    """
    try:
        # Initialize PDF reader
        pdf_reader = PdfReader(pdf_path)
        results = []
        
        # Iterate through each page
        for page_num in range(len(pdf_reader.pages)):
            # Extract text from the current page
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # Split text into sentences
            # This regex splits on .!? followed by spaces and capital letters
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            # Search for the word in each sentence
            for sentence in sentences:
                # Case-insensitive search
                if search_word.lower() in sentence.lower():
                    # Add tuple of (page_number, sentence) to results
                    # Using 1-based page numbering for better readability
                    results.append((page_num + 1, sentence.strip()))
        
        return results
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The PDF file '{pdf_path}' was not found.")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def main():
    """
    Example usage of the find_sentences_with_word function.
    """
    try:
        # Example usage
        pdf_path = "example.pdf"
        search_word = "Python"
        
        results = find_sentences_with_word(pdf_path, search_word)
        
        if results:
            print(f"\nFound {len(results)} sentences containing '{search_word}':")
            for page_num, sentence in results:
                print(f"\nPage {page_num}:")
                print(f"Sentence: {sentence}")
        else:
            print(f"No sentences containing '{search_word}' were found.")
            
    except Exception as e:
        print(f"Error: {str(e)}")