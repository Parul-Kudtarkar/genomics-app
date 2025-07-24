#!/usr/bin/env python3
"""
Setup environment for enhanced PDF ingestion pipeline
"""
import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment for enhanced PDF ingestion pipeline"""
    
    print("🔧 Setting up Enhanced PDF Ingestion Pipeline Environment")
    print("=" * 60)
    
    # Install requirements
    print("📦 Installing Python packages...")
    packages = [
        "PyPDF2==3.0.1",
        "pdfplumber==0.10.3", 
        "pinecone-client==2.2.4",
        "openai==1.3.5",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "numpy==1.24.3"
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        os.system(f"{sys.executable} -m pip install {package}")
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = ['logs', 'scripts', 'data', 'reports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created: {directory}/")
    
    # Create .env template
    env_template = """# Enhanced PDF Ingestion Pipeline Configuration
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration  
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=genomics-publications

# Enhanced Metadata Settings
EXTRACT_PDF_METADATA=true
ENRICH_WITH_CROSSREF=true
CROSSREF_EMAIL=your-email@domain.com

# Optional: Rate Limiting
CROSSREF_DELAY=0.5
MAX_CONCURRENT_REQUESTS=5
"""
    
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write(env_template)
        print("📝 Created .env template - please add your API keys")
    else:
        print("📝 .env file already exists")
    
    # Create example usage script
    example_script = """#!/usr/bin/env python3
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
"""
    
    with open('example_usage.py', 'w') as f:
        f.write(example_script)
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your actual API keys")
    print("2. Run: python enhanced_pdf_pipeline.py /path/to/pdfs sk-key... pin-key... index-name")
    print("3. Test search: python scripts/test_enhanced_search.py")
    print("4. View analytics: python scripts/analytics.py")
    
    print("\n💡 Example commands:")
    print("# Process PDFs with enhanced metadata")
    print("python enhanced_pdf_pipeline.py ./pdfs $OPENAI_API_KEY $PINECONE_API_KEY genomics-publications")
    print("\n# Interactive search")
    print("python scripts/test_enhanced_search.py --interactive")
    print("\n# Single query with filters")
    print("python scripts/test_enhanced_search.py --query 'CRISPR' --journal 'Nature' --year '2020-2024'")
    print("\n# Generate analytics report")
    print("python scripts/analytics.py --export")

if __name__ == "__main__":
    setup_environment()
