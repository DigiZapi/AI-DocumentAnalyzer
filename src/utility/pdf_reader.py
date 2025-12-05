import os
import sys
from io import BytesIO

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from config import MIN_IMAGE_SIZE, MAX_IMAGE_SIZE
except ImportError:
    # Fallback if config not available
    MIN_IMAGE_SIZE = 1000  # 1 KB
    MAX_IMAGE_SIZE = 500_000  # 500 KB


class PDFReader:
    
    def __init__(self, file_path: str, image_dir: str):
        self.file_path = file_path
        self.image_dir = image_dir

    
    def extract_images_with_pymupdf(self):
        """Use PyMuPDF (fitz) to extract images, including diagrams and vector graphics.
        
        Filters out very small images (icons, bullets) and very large images
        to prevent processing issues and reduce storage.
        """
        image_infos = []
        skipped_small = 0
        skipped_large = 0
        saved = 0
        
        try:
            doc = fitz.open(self.file_path)
            
            for p_idx in range(len(doc)):
                page = doc[p_idx]
                page_num = p_idx + 1
                
                # Get all images from the page
                image_list = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # image reference number
                        
                        # Extract the image
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Filter by size - skip tiny images (icons) and huge images
                        img_size = len(image_bytes)
                        
                        if img_size < MIN_IMAGE_SIZE:
                            skipped_small += 1
                            continue  # Skip small images (icons, bullets, UI elements)
                        
                        if img_size > MAX_IMAGE_SIZE:
                            skipped_large += 1
                            continue  # Skip very large images
                        
                        # Save the image
                        img_path = os.path.join(
                            self.image_dir,
                            f"{os.path.basename(self.file_path)}_p{page_num}_fitz_img{img_idx+1}.{image_ext}",
                        )
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        image_infos.append(
                            {
                                "id": f"{os.path.basename(self.file_path)}_fitz_img_p{page_num}_{img_idx+1}",
                                "page": page_num,
                                "path": img_path,
                                "source": os.path.basename(self.file_path),
                            }
                        )
                        saved += 1
                        
                    except Exception as e:
                        print(f"[warn] PyMuPDF: Error extracting image {img_idx} on page {page_num}: {e}")
            doc.close()
            
            # Print summary
            total = saved + skipped_small + skipped_large
            print(f"[info] Image extraction summary:")
            print(f"       Total images found: {total}")
            print(f"       ✅ Saved: {saved}")
            print(f"       ⏭️  Skipped (too small < {MIN_IMAGE_SIZE} bytes): {skipped_small}")
            print(f"       ⏭️  Skipped (too large > {MAX_IMAGE_SIZE} bytes): {skipped_large}")
            
        except Exception as e:
            print(f"[warn] PyMuPDF extraction failed: {e}")
        
        return image_infos
    
    
    
    def extract_pdf(self):
        """Extrahiere Text, Tabellen und Bilder aus einer PDF.

        Returns:
            text_chunks: [{"id", "page", "text", "source"}, ...]
            image_infos: [{"id", "page", "path", "source"}, ...]
            table_infos: [{"id", "page", "json", "df"}, ...]
        """

        os.makedirs(self.image_dir, exist_ok=True)

        text_chunks = []
        table_infos = []
        image_infos = []

        print(f"[info] Extracting PDF: {self.file_path}")
        
        with pdfplumber.open(self.file_path) as pdf:
            print(f"[info] Extracting text...")
            for p_idx, page in enumerate(pdf.pages):
                page_num = p_idx + 1
                text = (page.extract_text() or "").strip()
                if text:
                    text_chunks.append(
                        {
                            "id": f"{os.path.basename(self.file_path)}_text_p{page_num}",
                            "page": page_num,
                            "text": text,
                            "source": os.path.basename(self.file_path),
                        }
                    )

                print(f"[info] Extracting tables...")
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table:
                        continue

                    # Simple string preview; actual JSON/DF handled in rag_backend
                    table_text = "\n".join(
                        ["\t".join([str(cell) if cell else "" for cell in row]) for row in table]
                    )

                    table_infos.append(
                        {
                            "id": f"{os.path.basename(self.file_path)}_table_p{page_num}_{t_idx+1}",
                            "page": page_num,
                            "table": table,
                            "preview": table_text,
                            "source": os.path.basename(self.file_path),
                        }
                    )
        
        # PyMuPDF for better image/diagram detection
        print(f"[info] Extracting images with PyMuPDF for better diagram detection...")
        image_infos.extend(self.extract_images_with_pymupdf())

        return text_chunks, image_infos, table_infos



def extract_pdf(file_path: str):
    """Convenience wrapper matching existing app.py usage."""

    reader = PDFReader(file_path, image_dir="extracted_images")
    return reader.extract_pdf()
