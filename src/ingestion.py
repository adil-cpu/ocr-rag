import pdfplumber
import os
from typing import List, Dict, Union
from PIL import Image
import io

class PDFProcessor:
    """
    Handles PDF ingestion: extracts text and images.
    """
    def __init__(self, output_dir: str = "data/extracted"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def process_pdf(self, pdf_path: str) -> Dict[str, Union[List[str], List[str]]]:
        """
        Extracts text and saves images from a PDF.
        Returns a dictionary with 'text' chunks and 'image_paths'.
        """
        extracted_data = {
            "text": [],
            "image_paths": [],
            "metadata": {"source": pdf_path}
        }

        print(f"Processing: {pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    # 1. Extract Text
                    text = page.extract_text()
                    if text:
                        # Clean and chunk text (simple paragraph split for now)
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for p in paragraphs:
                            extracted_data["text"].append({
                                "content": p,
                                "page": page_num,
                                "type": "text"
                            })

                    # 2. Extract Images
                    # pdfplumber extracts image metadata
                    for img_index, img_obj in enumerate(page.images):
                        try:
                            # Get image coordinates
                            x0, top, x1, bottom = img_obj["x0"], img_obj["top"], img_obj["x1"], img_obj["bottom"]
                            # Crop the image from the page
                            cropped_page = page.within_bbox((x0, top, x1, bottom))
                            img_file = cropped_page.to_image()
                            
                            # Save image
                            image_filename = f"{os.path.basename(pdf_path)}_p{page_num}_img{img_index}.png"
                            image_path = os.path.join(self.images_dir, image_filename)
                            img_file.save(image_path)
                            
                            extracted_data["image_paths"].append({
                                "path": image_path,
                                "page": page_num,
                                "type": "image"
                            })
                        except Exception as e:
                            print(f"Failed to extract image on page {page_num}: {e}")

        except Exception as e:
            print(f"Error opening PDF: {e}")
            return None

        return extracted_data

if __name__ == "__main__":
    # Test run
    processor = PDFProcessor()
    # Dummy creation for testing if no PDF exists
    if not os.path.exists("test.pdf"):
        print("No test.pdf found. Place a PDF in the root to test.")
    else:
        data = processor.process_pdf("test.pdf")
        print(f"Extracted {len(data['text'])} text chunks and {len(data['image_paths'])} images.")
