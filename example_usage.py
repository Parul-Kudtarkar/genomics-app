#!/usr/bin/env python3
# Example usage of enhanced PDF pipeline
from enhanced_pdf_pipeline import EnhancedPDFPipeline

# Initialize pipeline
pipeline = EnhancedPDFPipeline(
    openai_key="your_openai_key",
    pinecone_key="your_pinecone_key", 
    index_name="genomics-publications"
)

# Process PDFs
stats = pipeline.process_folder("/path/to/your/pdf/folder")
print(f"Processed: {stats['processed']} files")

# Get summary
summary = pipeline.get_processing_summary()
print(summary)
