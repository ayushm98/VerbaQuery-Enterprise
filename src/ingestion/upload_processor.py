"""
Upload Processor for Dynamic PDF Uploads.

Processes uploaded PDFs using temporary files - no permanent storage.
Documents are added to indexes for immediate querying.
"""

import tempfile
from pathlib import Path
from typing import Dict, Callable, Optional
from config import get_settings
from src.utils import get_logger
from .pdf_loader import PDFLoader
from .chunker import SemanticChunker
from .incremental_indexer import IncrementalIndexer


logger = get_logger(__name__)


class UploadProcessor:
    """
    Process uploaded PDFs through the ingestion pipeline.
    Uses temporary files - PDFs are not stored permanently.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logger
        self.pdf_loader = PDFLoader()
        self.chunker = SemanticChunker()

    def process_uploaded_file(
        self,
        uploaded_file,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict:
        """
        Process a single uploaded PDF through the pipeline.

        Args:
            uploaded_file: Streamlit UploadedFile object
            progress_callback: Optional callback for progress updates
                              Takes (stage_name, progress_percent)

        Returns:
            dict with processing stats:
            {
                "success": bool,
                "filename": str,
                "pages": int,
                "chunks": int,
                "error": str (if failed)
            }
        """
        filename = uploaded_file.name

        self.logger.info(f"Processing uploaded file: {filename}")

        def update_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        try:
            # Step 1: Validate (0-10%)
            update_progress("Validating file...", 0.05)
            is_valid, error_msg = self.validate_upload(uploaded_file)
            if not is_valid:
                return {"success": False, "filename": filename, "error": error_msg}

            update_progress("Validation complete", 0.10)

            # Step 2: Save to temporary file (10-20%)
            update_progress("Preparing file...", 0.15)

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = Path(tmp_file.name)

            update_progress("File ready", 0.20)

            try:
                # Step 3: Load PDF (20-40%)
                update_progress("Loading PDF...", 0.25)
                documents = self.pdf_loader.load_single_pdf(temp_path)
                pages_count = len(documents)
                update_progress(f"Loaded {pages_count} pages", 0.40)

                if not documents:
                    return {
                        "success": False,
                        "filename": filename,
                        "error": "No content extracted from PDF"
                    }

                # Step 4: Chunk documents (40-60%)
                update_progress("Chunking documents...", 0.45)
                chunked_docs = self.chunker.chunk_documents(documents)
                chunks_count = len(chunked_docs)
                update_progress(f"Created {chunks_count} chunks", 0.60)

                # Step 5: Add to indexes (60-95%)
                update_progress("Adding to indexes...", 0.65)
                # Create fresh indexer for each upload to avoid stale ChromaDB connections
                indexer = IncrementalIndexer()
                index_stats = indexer.add_documents(chunked_docs)
                update_progress("Indexing complete", 0.95)

                # Done (100%)
                update_progress("Complete!", 1.0)

                return {
                    "success": True,
                    "filename": filename,
                    "pages": pages_count,
                    "chunks": chunks_count,
                    "index_stats": index_stats
                }

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
                    self.logger.info(f"Cleaned up temporary file: {temp_path}")

        except Exception as e:
            self.logger.error(f"Failed to process {filename}: {e}")
            return {
                "success": False,
                "filename": filename,
                "error": str(e)
            }

    def validate_upload(self, uploaded_file) -> tuple[bool, str]:
        """
        Validate uploaded file before processing.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            (is_valid, error_message)
        """
        filename = uploaded_file.name

        # Check file extension
        if not filename.lower().endswith('.pdf'):
            return False, "Only PDF files are supported"

        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > self.settings.max_upload_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({self.settings.max_upload_size_mb}MB)"

        # Check if file has content
        if uploaded_file.size == 0:
            return False, "File is empty"

        return True, ""
