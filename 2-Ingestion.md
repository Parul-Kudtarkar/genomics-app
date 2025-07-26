# 📚 **Document Ingestion Guide**

## 🎯 **Overview**

This guide covers ingesting documents into your genomics vector database using our specialized pipelines. We support multiple document formats with automatic processing and metadata extraction.

## 🚀 **Quick Start (5 minutes)**

### 1. **Install Dependencies**

```bash
# Install required packages
pip install openai==1.3.5 pinecone-client==2.2.4 python-dotenv==1.0.0 requests==2.31.0 numpy==1.24.3
```

### 2. **Configure Environment**

Create a `.env` file in your project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration  
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=genomics-publications
```

### 3. **Choose Your Pipeline**

#### **For Text Files (.txt)**
```bash
# Process a single text file
python text_ingestion_pipeline.py ingest-file "path/to/document.txt"

# Process all text files in a directory
python text_ingestion_pipeline.py ingest-dir "path/to/text/directory"

# Process multiple specific files
python text_ingestion_pipeline.py ingest-files "file1.txt" "file2.txt" "file3.txt"
```

#### **For XML Files (.xml)**
```bash
# Process a single XML file (PubMed, arXiv, bioRxiv, PMC)
python xml_ingestion_pipeline.py ingest-file "path/to/paper.xml"

# Process all XML files in a directory
python xml_ingestion_pipeline.py ingest-dir "path/to/xml/directory"

# Process multiple specific XML files
python xml_ingestion_pipeline.py ingest-files "file1.xml" "file2.xml" "file3.xml"
```

## 📋 **Supported Formats**

### **Text Pipeline** (`text_ingestion_pipeline.py`)
- 📄 **Plain text files** (.txt)
- 📝 **Converted PDFs** (text extracted from PDFs)
- 📋 **Research notes** and documents
- 📊 **Data files** with text content

### **XML Pipeline** (`xml_ingestion_pipeline.py`)
- 🔬 **PubMed XML** - Medical and scientific papers
- 📚 **arXiv XML** - Preprints and research papers
- 🧬 **bioRxiv XML** - Biology preprints
- 📖 **PMC XML** - PubMed Central articles (JATS format)
- 🔍 **Generic XML** - Any academic paper XML format

## 🔄 **PDF Conversion Process**

### **Professional PDF Converter**

For PDF documents, we provide a professional conversion tool that transforms PDFs into structured text before ingestion:

#### **Conversion Methods**

The PDF converter uses two advanced parsing methods:

1. **GROBID (Primary)** - High-quality academic document parser
   - Extracts structured content (title, authors, abstract, sections)
   - Handles complex academic layouts
   - Preserves document structure and metadata
   - Requires GROBID server running locally

2. **Unstructured (Fallback)** - Robust general-purpose parser
   - Works with any PDF format
   - Extracts text content and basic structure
   - No additional server requirements
   - Handles various document types

#### **Extracted Content**

The converter extracts comprehensive document information:

- **Title** - Document title and main heading
- **Authors** - Author names and affiliations
- **Abstract** - Document abstract or summary
- **Full Text** - Complete document content
- **Sections** - Structured sections with headings
- **References** - Bibliography and citations
- **Metadata** - Publication information and identifiers

#### **Output Formats**

Converted PDFs can be saved in two formats:

1. **Structured Text** - Human-readable formatted text
   - Clear section separation
   - Preserved document structure
   - Easy to review and edit
   - Ready for text ingestion pipeline

2. **JSON** - Machine-readable structured data
   - Complete metadata preservation
   - Programmatic access to all content
   - Custom processing capabilities
   - Integration with other systems

#### **Usage Examples**

```bash
# Convert a single PDF to text
python professional_pdf_converter.py convert-file "research_paper.pdf"

# Convert with custom output directory
python professional_pdf_converter.py convert-file "paper.pdf" --output-dir "./converted/"

# Convert to JSON format
python professional_pdf_converter.py convert-file "paper.pdf" --format json

# Convert all PDFs in a directory
python professional_pdf_converter.py convert-dir "./pdf_papers/"

# Convert all PDFs with custom output
python professional_pdf_converter.py convert-dir "./pdfs/" --output-dir "./text_files/" --format text
```

#### **Conversion Workflow**

The complete PDF processing workflow:

1. **PDF Input** - Original PDF documents
2. **Conversion** - PDF to structured text/JSON
3. **Validation** - Quality check of converted content
4. **Text Ingestion** - Process converted text files
5. **Vector Storage** - Store in vector database

#### **Quality Assurance**

The converter includes several quality checks:

- **Content Validation** - Ensures sufficient text was extracted
- **Structure Preservation** - Maintains document organization
- **Metadata Extraction** - Captures publication information
- **Fallback Handling** - Uses alternative methods if primary fails
- **Error Reporting** - Detailed logging of conversion issues

## 🔧 **Configuration Options**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key for embeddings |
| `PINECONE_API_KEY` | Required | Pinecone API key for vector storage |
| `PINECONE_INDEX_NAME` | `genomics-publications` | Target vector database index |

### **Command Line Arguments**

Both pipelines support these optional arguments:
```bash
--openai-key "your_key"      # Override environment variable
--pinecone-key "your_key"    # Override environment variable  
--index-name "your_index"    # Override environment variable
```

## 📊 **What Gets Extracted**

### **Text Pipeline Features**
- 📄 **Content** - Full text content
- 🏷️ **Title** - Auto-extracted from filename or content
- 📅 **Metadata** - File creation date, size, path
- 🆔 **Unique ID** - Hash-based document identifier
- 🧩 **Chunks** - Semantic text segments (2000 chars, 200 overlap)

### **XML Pipeline Features**
- 📄 **Complete Articles** - Full text extraction
- 👥 **Authors** - Author names and affiliations
- 📰 **Journal** - Publication venue
- 📅 **Publication Date** - Year and date information
- 🔗 **DOI** - Digital Object Identifier
- 📝 **Abstract** - Paper abstract
- 🔤 **Keywords** - Subject keywords
- 🌐 **URL** - Publication URL
- 📋 **License** - Publication license
- 🧩 **Chunks** - Semantic text segments (2000 chars, 200 overlap)

## 🚀 **Usage Examples**

### **PDF Processing Workflow**

```bash
# Step 1: Convert PDFs to text
python professional_pdf_converter.py convert-dir "./research_papers/" --output-dir "./converted_text/"

# Step 2: Ingest converted text files
python text_ingestion_pipeline.py ingest-dir "./converted_text/"
```

### **Text File Processing**

```bash
# Process a single research document
python text_ingestion_pipeline.py ingest-file "./research_notes.txt"

# Process all text files in research directory
python text_ingestion_pipeline.py ingest-dir "./research_papers/"

# Process specific converted PDFs
python text_ingestion_pipeline.py ingest-files \
  "paper1_converted.txt" \
  "paper2_converted.txt" \
  "notes.txt"
```

### **XML File Processing**

```bash
# Process PubMed XML export
python xml_ingestion_pipeline.py ingest-file "./pubmed_export.xml"

# Process arXiv papers
python xml_ingestion_pipeline.py ingest-file "./arxiv_papers.xml"

# Process all XML files in directory
python xml_ingestion_pipeline.py ingest-dir "./xml_papers/"

# Process multiple specific XML files
python xml_ingestion_pipeline.py ingest-files \
  "pubmed_export.xml" \
  "biorxiv_papers.xml" \
  "pmc_articles.xml"
```

### **Batch Processing**

```bash
# Process multiple directories
for dir in ./text_papers_2020 ./text_papers_2021 ./text_papers_2022; do
  echo "Processing $dir..."
  python text_ingestion_pipeline.py ingest-dir "$dir"
done

# Process XML files by type
for xml_file in ./pubmed/*.xml ./arxiv/*.xml ./biorxiv/*.xml; do
  python xml_ingestion_pipeline.py ingest-file "$xml_file"
done
```

## 📈 **Monitoring Progress**

### **Real-time Logs**

Both pipelines provide detailed logging:

```
2025-01-15 10:30:15 - INFO - 🚀 Text Ingestion Pipeline initialized
2025-01-15 10:30:15 - INFO -    📊 Target index: genomics-publications
2025-01-15 10:30:16 - INFO - ✅ OpenAI client initialized successfully
2025-01-15 10:30:16 - INFO - ✅ Pinecone initialized - 1,250 vectors
2025-01-15 10:30:16 - INFO - 📄 Ingesting text file: research_notes.txt
2025-01-15 10:30:17 - INFO - 📄 Processing text: Research Notes on CRISPR Gene Editing...
2025-01-15 10:30:18 - INFO - 🤖 Generating embeddings for 3 texts...
2025-01-15 10:30:19 - INFO - ✅ Generated 3 embeddings
2025-01-15 10:30:20 - INFO - ✅ Processed successfully (3 chunks)
```

### **Processing Summary**

After completion, you'll see:

```
📊 INGESTION SUMMARY:
   📄 File: research_notes.txt
   ✅ Processed: True
   🧩 Chunks created: 3
   📊 Vectors uploaded: 3

📊 DIRECTORY INGESTION SUMMARY:
   📁 Directory: ./xml_papers/
   📄 Files found: 15
   ✅ Files processed: 14
   ❌ Files failed: 1
   🧩 Total chunks: 127
```

## 🔍 **Testing Your Ingestion**

### **Check Processing History**

```bash
# View processed text files
cat processed_text_files.json | jq '.[] | {title, source, chunk_count}'

# View processed XML files  
cat processed_xml_files.json | jq '.[] | {title, source, chunk_count}'
```

### **Run Analytics**

```bash
# View vector store analytics
python scripts/analytics.py

# Export detailed report
python scripts/analytics.py --export
```

### **Test Search**

```bash
# Interactive search mode
python scripts/test_enhanced_search.py --interactive

# Single query with filters
python scripts/test_enhanced_search.py \
  --query "CRISPR gene editing" \
  --year "2020-2024"
```

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **"OPENAI_API_KEY required"**
```bash
# Check your API key
echo $OPENAI_API_KEY | head -c 10
# Should start with "sk-"

# Verify in .env file
cat .env | grep OPENAI_API_KEY
```

#### **"PINECONE_API_KEY required"**
```bash
# Verify Pinecone key format
echo $PINECONE_API_KEY | head -c 10
# Should start with "pc-"
```

#### **"Could not initialize Pinecone"**
```bash
# Check if index exists
python -c "
from pinecone import Pinecone
pc = Pinecone(api_key='your-key')
print(pc.list_indexes())
"
```

#### **"XML parsing failed"**
- Ensure XML files are well-formed
- Check for encoding issues (should be UTF-8)
- Verify XML format is supported (PubMed, arXiv, etc.)

#### **"Text file is empty"**
- Check file encoding (should be UTF-8)
- Ensure file contains actual text content
- Verify file permissions

#### **"PDF conversion failed"**
- Ensure PDFs are not password-protected
- Check if PDFs contain actual text (not just images)
- Verify GROBID server is running (if using GROBID)
- Check Unstructured installation (if using fallback)

### **Performance Optimization**

#### **For Large Collections**
```bash
# Process in smaller batches
find ./xml_papers -name "*.xml" | split -l 10 - batch_
for batch in batch_*; do
  python xml_ingestion_pipeline.py ingest-files $(cat $batch)
  sleep 2  # Rate limiting
done
```

#### **Memory Management**
```bash
# Reduce chunk size for memory-constrained systems
# Edit the create_chunks method in both pipelines:
chunk_size = 1000  # Default is 2000
overlap = 100      # Default is 200
```

## 📊 **Analytics & Reporting**

### **View Processing History**

```bash
# Check processed text files
cat processed_text_files.json | jq '.[] | {title, source, chunk_count, processed_at}'

# Check processed XML files
cat processed_xml_files.json | jq '.[] | {title, source, chunk_count, processed_at}'
```

### **Generate Reports**

```bash
# Comprehensive analytics
python scripts/analytics.py --export

# This creates: analytics_report_20250115_143022.json
```

### **Sample Analytics Output**

```json
{
  "total_vectors": 1250,
  "metadata_coverage": {
    "journals": "78.5%",
    "authors": "92.3%", 
    "dois": "65.2%",
    "citations": "45.8%"
  },
  "top_journals": {
    "Nature": 45,
    "Science": 32,
    "Cell": 28
  },
  "year_distribution": {
    "2024": 156,
    "2023": 234,
    "2022": 198
  }
}
```

## 🔄 **Data Migration**

### **Migrate Existing Data**

If you have existing vectors without enhanced metadata:

```bash
# Analyze current metadata
python scripts/migrate_existing_data.py --analyze

# Migrate to enhanced format
python scripts/migrate_existing_data.py
```

## 📚 **Advanced Features**

### **Custom XML Parsing**

You can extend the XML pipeline for custom formats:

```python
# Add to xml_ingestion_pipeline.py
def parse_custom_xml(self, xml_content: str) -> List[PaperRecord]:
    """Parse custom XML format"""
    papers = []
    try:
        root = ET.fromstring(xml_content)
        
        # Add your custom parsing logic here
        # Example: Extract from custom tags
        for paper_elem in root.findall('.//custom_paper'):
            title = paper_elem.find('.//title').text
            authors = [author.text for author in paper_elem.findall('.//author')]
            # ... more extraction logic
            
            papers.append(PaperRecord(
                source='custom',
                id=paper_id,
                title=title,
                authors=authors,
                # ... other fields
            ))
    
    except Exception as e:
        logger.error(f"Custom XML parsing failed: {e}")
    
    return papers
```

### **Batch Processing Script**

Create a custom batch processor:

```bash
#!/bin/bash
# batch_process.sh

TYPE=$1  # "text" or "xml"
FOLDER=$2

echo "Processing $TYPE files in $FOLDER..."

if [ "$TYPE" = "text" ]; then
    python text_ingestion_pipeline.py ingest-dir "$FOLDER"
elif [ "$TYPE" = "xml" ]; then
    python xml_ingestion_pipeline.py ingest-dir "$FOLDER"
else
    echo "Invalid type. Use 'text' or 'xml'"
    exit 1
fi
```

### **Automated Processing**

Set up automated ingestion:

```bash
#!/bin/bash
# auto_ingest.sh

# Watch for new files and process them
inotifywait -m -e create,moved_to /path/to/watch/directory |
while read path action file; do
    if [[ "$file" =~ \.(txt|xml)$ ]]; then
        echo "New file detected: $file"
        
        if [[ "$file" =~ \.txt$ ]]; then
            python text_ingestion_pipeline.py ingest-file "$path$file"
        elif [[ "$file" =~ \.xml$ ]]; then
            python xml_ingestion_pipeline.py ingest-file "$path$file"
        fi
    fi
done
```

## 🎯 **Next Steps**

After successful ingestion:

1. **Test Search**: Use the search testing scripts
2. **Build UI**: Connect to your frontend application
3. **Monitor**: Set up analytics and monitoring
4. **Scale**: Optimize for larger collections

## 📞 **Support**

- 📖 **Documentation**: Check other guides in this series
- 🐛 **Issues**: Review logs in `text_ingestion_pipeline.log` or `xml_ingestion_pipeline.log`
- 🔧 **Debug**: Check processed files JSON for tracking

---

**Ready to start?** Choose your pipeline and begin ingesting your documents! 🚀