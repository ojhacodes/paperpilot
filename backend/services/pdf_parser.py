import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional

class PDFParser:
    def __init__(self):
        # Regexes for header detection
        self.section_patterns = [
            # Matches "1. Introduction", "1 Introduction", "1.1 Related Work", etc.
            re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9\s,\-\:\(\)]+)$'),
            # Matches "I. INTRODUCTION", "II. RELATED WORK", etc.
            re.compile(r'^([I|V|X|L|C]+)\.\s+([A-Z\s,\-\:\(\)]+)$'),
            # Matches stand-alone major sections like "Abstract", "Introduction", "References", "Appendix"
            re.compile(r'^(Abstract|Introduction|Background|Methodology|Methods|Model|Architecture|Experiments|Evaluation|Results|Discussion|Related\s+Work|Conclusion|Conclusions|References|Appendices|Appendix)$', re.IGNORECASE)
        ]

    def parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of pages with their text,
        page numbers, and detected section headers.
        """
        doc = fitz.open(file_path)
        pages_data = []
        current_section = "Abstract"  # Start with Abstract by default
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            # Extract text blocks
            # Each block: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            
            page_text_blocks = []
            for block in blocks:
                text = block[4].strip()
                if not text:
                    continue
                
                # Check if this block looks like a section header
                # We split by newline and look at the first line of the block
                first_line = text.split("\n")[0].strip()
                
                # Filter out headers that are too long to be section names
                is_header = False
                if len(first_line) < 100:
                    for pattern in self.section_patterns:
                        if pattern.match(first_line):
                            current_section = first_line
                            is_header = True
                            break
                
                page_text_blocks.append({
                    "text": text,
                    "section": current_section,
                    "is_header": is_header
                })
            
            pages_data.append({
                "page_number": page_num,
                "blocks": page_text_blocks,
                "full_text": page.get_text()
            })
            
        doc.close()
        return pages_data

    def chunk_pages(
        self, 
        pages_data: List[Dict[str, Any]], 
        chunk_size_tokens: int = 512, 
        overlap_tokens: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Chunks the parsed page data using a sliding window.
        Approximate 1 word = 1.3 tokens (so chunk_size in words = chunk_size_tokens / 1.3).
        """
        # Convert tokens to approximate word counts
        chunk_size_words = int(chunk_size_tokens / 1.3)
        overlap_words = int(overlap_tokens / 1.3)
        
        if chunk_size_words <= overlap_words:
            overlap_words = chunk_size_words // 2

        chunks = []
        
        # Flatten all blocks into a list of words with metadata
        words_metadata = []
        for page in pages_data:
            page_num = page["page_number"]
            for block in page["blocks"]:
                section = block["section"]
                text = block["text"]
                # Split text into words while keeping track of page/section
                words = text.split()
                for word in words:
                    words_metadata.append({
                        "word": word,
                        "page_number": page_num,
                        "section": section
                    })

        total_words = len(words_metadata)
        if total_words == 0:
            return chunks

        i = 0
        chunk_idx = 0
        while i < total_words:
            # Get the slice of words for this chunk
            chunk_end = min(i + chunk_size_words, total_words)
            chunk_slice = words_metadata[i:chunk_end]
            
            if not chunk_slice:
                break
                
            # Reconstruct chunk text
            chunk_text = " ".join([item["word"] for item in chunk_slice])
            
            # Determine pages covered by this chunk
            pages_in_chunk = sorted(list(set([item["page_number"] for item in chunk_slice])))
            sections_in_chunk = sorted(list(set([item["section"] for item in chunk_slice])))
            
            # Use the page and section of the middle word to represent the chunk
            mid_item = chunk_slice[len(chunk_slice) // 2]
            primary_page = mid_item["page_number"]
            primary_section = mid_item["section"]
            
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "primary_page": primary_page,
                "primary_section": primary_section,
                "pages_covered": pages_in_chunk,
                "sections_covered": sections_in_chunk,
            })
            
            chunk_idx += 1
            
            # Move index forward by step (chunk_size - overlap)
            step = chunk_size_words - overlap_words
            if step <= 0:
                step = 1
            i += step
            
            # Avoid infinite loop or tiny steps if end is reached
            if i >= total_words or chunk_end == total_words:
                break
                
        return chunks

# Test code
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        data = parser.parse_pdf(sys.argv[1])
        chunks = parser.chunk_pages(data, 256, 50)
        print(f"Parsed {len(data)} pages. Created {len(chunks)} chunks.")
        if chunks:
            print("First chunk:")
            print(chunks[0])
