from pathlib import Path
from typing import List, Dict
import pypdf
from langchain_core.documents import Document
from src.utils import get_logger, validate_pdf_file


logger = get_logger(__name__)


class PDFLoader:
    """
    Load and extract text from PDF files with metadata preservation.
    """

    def __init__(self):
        self.logger = logger

    def load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a single PDF file and extract text with page-level metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects, one per page
        """
        validate_pdf_file(pdf_path)

        documents = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                self.logger.info(f"Loading {pdf_path.name}: {total_pages} pages")

                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()

                    # Skip empty pages
                    if not text.strip():
                        self.logger.warning(f"Skipping empty page {page_num} in {pdf_path.name}")
                        continue

                    # Create Document with rich metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "file_path": str(pdf_path.absolute())
                        }
                    )
                    documents.append(doc)

                self.logger.info(f"Extracted {len(documents)} non-empty pages from {pdf_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to load {pdf_path}: {str(e)}")
            raise

        return documents

    def load_directory(self, directory_path: Path) -> List[Document]:
        """
        Load all PDF files from a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of all Document objects from all PDFs
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(directory_path.glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory_path}")
            return []

        self.logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")

        all_documents = []

        for pdf_file in pdf_files:
            try:
                docs = self.load_single_pdf(pdf_file)
                all_documents.extend(docs)
            except Exception as e:
                self.logger.error(f"Skipping {pdf_file.name} due to error: {str(e)}")
                continue

        self.logger.info(f"Total documents loaded: {len(all_documents)}")

        return all_documents
