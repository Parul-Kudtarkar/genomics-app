## 🚀 **Quick Setup (2 minutes):**

```bash
# 1. Save the pipeline file
# Copy the main code to: pdf_ingestion_pipeline.py

"""
Enhanced PDF Processing Pipeline with Rich Metadata Extraction
Builds on the original simple pipeline with metadata enhancement capabilities
"""
import os
import sys
import hashlib
import json
import logging
import re
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# PDF processing
import PyPDF2
import pdfplumber

# AI libraries
import openai

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_pdf_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPDFPipeline:
    """Enhanced PDF processing pipeline with rich metadata extraction"""
    
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str):
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.index_name = index_name
        
        # Configuration
        self.crossref_email = os.getenv('CROSSREF_EMAIL', 'genomics-app@example.com')
        self.enable_crossref = os.getenv('ENRICH_WITH_CROSSREF', 'true').lower() == 'true'
        self.extract_metadata = os.getenv('EXTRACT_PDF_METADATA', 'true').lower() == 'true'
        
        # Initialize clients
        self.openai_client = self._init_openai()
        self.pinecone_index = self._init_pinecone()
        
        # Load processed files tracking
        self.processed_file = 'processed_files.json'
        self.processed_files = self._load_processed_files()
        
        # Metadata cache for session
        self.metadata_cache = {}
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            # Try new OpenAI client first
            client = openai.OpenAI(api_key=self.openai_key)
            # Quick test
            try:
                client.models.list()
                logger.info("✅ OpenAI client (new) initialized successfully")
                return client
            except:
                # If test fails, fall back to legacy
                openai.api_key = self.openai_key
                logger.info("✅ OpenAI client (legacy) initialized successfully")
                return None
        except Exception as e:
            # Legacy initialization
            try:
                openai.api_key = self.openai_key
                logger.info("✅ OpenAI client (legacy) initialized successfully")
                return None
            except Exception as e2:
                logger.error(f"OpenAI initialization failed: {e}, {e2}")
                raise Exception(f"Could not initialize OpenAI: {e}")
    
    def _init_pinecone(self):
        """Initialize Pinecone client"""
        try:
            # Try new Pinecone client
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_key)
            index = pc.Index(self.index_name)
            logger.info("✅ Pinecone client (new) initialized successfully")
            return index
        except Exception as e:
            # Try legacy Pinecone
            try:
                import pinecone
                pinecone.init(api_key=self.pinecone_key)
                index = pinecone.Index(self.index_name)
                logger.info("✅ Pinecone client (legacy) initialized successfully")
                return index
            except Exception as e2:
                logger.error(f"Pinecone initialization failed: {e}, {e2}")
                raise Exception(f"Could not initialize Pinecone: {e}")
    
    def _load_processed_files(self) -> Dict[str, Any]:
        """Load processed files tracking"""
        try:
            if Path(self.processed_file).exists():
                with open(self.processed_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def _save_processed_files(self):
        """Save processed files tracking"""
        try:
            with open(self.processed_file, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed files: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file"""
        metadata = {
            'doi': None,
            'authors': [],
            'journal': None,
            'publication_year': None,
            'title': None,
            'keywords': [],
            'institutions': [],
            'abstract': None
        }
        
        if not self.extract_metadata:
            return metadata
        
        try:
            filename = Path(pdf_path).name
            logger.info(f"🔍 Extracting metadata from: {filename}")
            
            # Try pdfplumber first for metadata
            with pdfplumber.open(pdf_path) as pdf:
                pdf_metadata = pdf.metadata
                
                if pdf_metadata:
                    # Extract title
                    if pdf_metadata.get('Title'):
                        title = pdf_metadata['Title'].strip()
                        if len(title) > 10 and len(title) < 300:  # Reasonable title length
                            metadata['title'] = title
                    
                    # Extract author
                    if pdf_metadata.get('Author'):
                        authors_text = pdf_metadata['Author']
                        # Split by common separators and clean
                        authors = []
                        for separator in [',', ';', ' and ', ' & ']:
                            if separator in authors_text:
                                authors = [a.strip() for a in authors_text.split(separator) if a.strip()]
                                break
                        
                        if not authors:  # Single author
                            authors = [authors_text.strip()]
                        
                        # Clean and validate authors
                        clean_authors = []
                        for author in authors[:10]:  # Limit to 10 authors
                            author = author.strip()
                            if 3 < len(author) < 100:  # Reasonable author name length
                                clean_authors.append(author)
                        
                        metadata['authors'] = clean_authors
                    
                    # Extract creation date for year
                    if pdf_metadata.get('CreationDate'):
                        try:
                            creation_date = pdf_metadata['CreationDate']
                            if hasattr(creation_date, 'year'):
                                year = creation_date.year
                                if 1900 <= year <= datetime.now().year:
                                    metadata['publication_year'] = year
                        except:
                            pass
                
                # Extract information from first few pages
                first_pages_text = ""
                for i, page in enumerate(pdf.pages[:3]):  # First 3 pages
                    if page_text := page.extract_text():
                        first_pages_text += page_text + "\n"
                
                if first_pages_text:
                    # Extract DOI using regex
                    doi_patterns = [
                        r'doi[:\s]*([0-9]+\.[0-9]+\/[^\s\)]+)',
                        r'DOI[:\s]*([0-9]+\.[0-9]+\/[^\s\)]+)',
                        r'https?://doi\.org/([0-9]+\.[0-9]+\/[^\s\)]+)',
                        r'dx\.doi\.org/([0-9]+\.[0-9]+\/[^\s\)]+)'
                    ]
                    
                    for pattern in doi_patterns:
                        doi_match = re.search(pattern, first_pages_text, re.IGNORECASE)
                        if doi_match:
                            doi = doi_match.group(1).strip().rstrip('.')
                            if len(doi) > 7:  # Reasonable DOI length
                                metadata['doi'] = doi
                                logger.info(f"📄 Found DOI: {doi}")
                            break
                    
                    # Extract journal name
                    journal_patterns = [
                        r'Published in[:\s]+([^,\n\r]+)',
                        r'Journal of\s+([^,\n\r]+)',
                        r'©.*?\d{4}\s+([^,\n\r]+)',
                        r'Proceedings of[:\s]+([^,\n\r]+)',
                        r'In:\s+([^,\n\r]+)',
                        r'Source:\s+([^,\n\r]+)'
                    ]
                    
                    for pattern in journal_patterns:
                        journal_match = re.search(pattern, first_pages_text, re.IGNORECASE)
                        if journal_match:
                            potential_journal = journal_match.group(1).strip()
                            # Clean and validate journal name
                            potential_journal = re.sub(r'\s+', ' ', potential_journal)
                            if 5 < len(potential_journal) < 150:  # Reasonable journal name length
                                metadata['journal'] = potential_journal
                                logger.info(f"📰 Found journal: {potential_journal}")
                            break
                    
                    # Extract publication year from text
                    if not metadata['publication_year']:
                        year_patterns = [
                            r'©\s*(\d{4})',
                            r'Published[:\s]+.*?(\d{4})',
                            r'(\d{4})\s*©',
                            r'Copyright.*?(\d{4})'
                        ]
                        
                        for pattern in year_patterns:
                            year_match = re.search(pattern, first_pages_text)
                            if year_match:
                                year = int(year_match.group(1))
                                if 1900 <= year <= datetime.now().year:
                                    metadata['publication_year'] = year
                                    logger.info(f"📅 Found year: {year}")
                                break
                    
                    # Extract institutions/affiliations
                    institution_patterns = [
                        r'([A-Z][a-z]+ University)',
                        r'University of ([A-Z][^,\n\r]+)',
                        r'([A-Z][^,\n\r]+ Institute(?:\s+of[^,\n\r]+)?)',
                        r'([A-Z][^,\n\r]+ Hospital)',
                        r'([A-Z][^,\n\r]+ Medical Center)',
                        r'([A-Z][^,\n\r]+ Research Center)',
                        r'([A-Z][^,\n\r]+ Laboratory)'
                    ]
                    
                    institutions = set()
                    for pattern in institution_patterns:
                        matches = re.findall(pattern, first_pages_text)
                        for match in matches:
                            if isinstance(match, tuple):
                                institution = ' '.join(match).strip()
                            else:
                                institution = match.strip()
                            
                            # Clean and validate institution name
                            institution = re.sub(r'\s+', ' ', institution)
                            if 10 < len(institution) < 100:  # Reasonable length
                                institutions.add(institution)
                    
                    metadata['institutions'] = list(institutions)[:5]  # Limit to 5
                    
                    # Try to extract abstract
                    abstract_patterns = [
                        r'ABSTRACT[:\s]+(.*?)(?:KEYWORDS|INTRODUCTION|1\.|Keywords)',
                        r'Abstract[:\s]+(.*?)(?:Keywords|Introduction|1\.|KEYWORDS)',
                        r'Summary[:\s]+(.*?)(?:Keywords|Introduction|1\.|KEYWORDS)'
                    ]
                    
                    for pattern in abstract_patterns:
                        abstract_match = re.search(pattern, first_pages_text, re.IGNORECASE | re.DOTALL)
                        if abstract_match:
                            abstract = abstract_match.group(1).strip()
                            # Clean abstract
                            abstract = re.sub(r'\s+', ' ', abstract)
                            if 50 < len(abstract) < 2000:  # Reasonable abstract length
                                metadata['abstract'] = abstract
                                logger.info(f"📝 Found abstract: {len(abstract)} chars")
                            break
        
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata from {pdf_path}: {e}")
        
        return metadata
    
    def enrich_with_crossref(self, doi: str) -> Dict[str, Any]:
        """Enrich metadata using Crossref API"""
        if not doi or not self.enable_crossref:
            return {}
        
        # Check cache first
        if doi in self.metadata_cache:
            logger.info(f"🔄 Using cached Crossref data for {doi}")
            return self.metadata_cache[doi]
        
        try:
            logger.info(f"🌐 Enriching with Crossref: {doi}")
            
            url = f"https://api.crossref.org/works/{doi}"
            headers = {
                'User-Agent': f'GenomicsApp/1.0 (mailto:{self.crossref_email})'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})
                
                enriched = {
                    'crossref_title': None,
                    'crossref_journal': None,
                    'crossref_authors': [],
                    'crossref_year': None,
                    'citation_count': 0,
                    'publisher': None,
                    'issn': None,
                    'subject_areas': [],
                    'crossref_abstract': None
                }
                
                # Extract title
                if work.get('title'):
                    enriched['crossref_title'] = work['title'][0]
                
                # Extract journal
                if work.get('container-title'):
                    enriched['crossref_journal'] = work['container-title'][0]
                
                # Extract authors
                if work.get('author'):
                    authors = []
                    for author in work['author'][:15]:  # Limit to 15 authors
                        given = author.get('given', '')
                        family = author.get('family', '')
                        if family:  # At least last name required
                            full_name = f"{given} {family}".strip()
                            authors.append(full_name)
                    enriched['crossref_authors'] = authors
                
                # Extract publication year
                if work.get('published-print', {}).get('date-parts'):
                    year_parts = work['published-print']['date-parts'][0]
                    if year_parts and len(year_parts) > 0:
                        enriched['crossref_year'] = year_parts[0]
                elif work.get('published-online', {}).get('date-parts'):
                    year_parts = work['published-online']['date-parts'][0]
                    if year_parts and len(year_parts) > 0:
                        enriched['crossref_year'] = year_parts[0]
                
                # Extract citation count
                enriched['citation_count'] = work.get('is-referenced-by-count', 0)
                
                # Extract publisher
                enriched['publisher'] = work.get('publisher')
                
                # Extract ISSN
                if work.get('ISSN'):
                    enriched['issn'] = work['ISSN'][0]
                
                # Extract subject areas
                if work.get('subject'):
                    enriched['subject_areas'] = work['subject'][:10]  # Limit to 10
                
                # Extract abstract if available
                if work.get('abstract'):
                    abstract = work['abstract']
                    # Clean HTML tags if present
                    abstract = re.sub(r'<[^>]+>', '', abstract)
                    if len(abstract) > 50:
                        enriched['crossref_abstract'] = abstract
                
                # Cache the result
                self.metadata_cache[doi] = enriched
                
                logger.info(f"✅ Crossref enriched: journal={enriched['crossref_journal']}, "
                           f"authors={len(enriched['crossref_authors'])}, "
                           f"citations={enriched['citation_count']}")
                
                return enriched
            
            elif response.status_code == 404:
                logger.warning(f"DOI not found in Crossref: {doi}")
            else:
                logger.warning(f"Crossref API returned status {response.status_code} for DOI {doi}")
        
        except requests.exceptions.Timeout:
            logger.warning(f"Crossref API timeout for DOI {doi}")
        except Exception as e:
            logger.warning(f"Crossref enrichment failed for {doi}: {e}")
        
        return {}
    
    def _extract_simple_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text"""
        # Expanded scientific/medical keywords
        science_keywords = [
            # Molecular biology
            'CRISPR', 'gene editing', 'RNA', 'DNA', 'protein', 'sequencing',
            'genomics', 'transcriptomics', 'proteomics', 'metabolomics',
            'single-cell', 'scRNA-seq', 'ChIP-seq', 'ATAC-seq',
            
            # AI/ML
            'machine learning', 'artificial intelligence', 'deep learning',
            'neural network', 'algorithm', 'bioinformatics',
            
            # Medical
            'cancer', 'tumor', 'therapy', 'treatment', 'diagnosis', 'biomarker',
            'clinical trial', 'drug discovery', 'personalized medicine',
            'immunotherapy', 'chemotherapy', 'radiotherapy',
            
            # Research methods
            'PCR', 'Western blot', 'immunofluorescence', 'flow cytometry',
            'mass spectrometry', 'NMR', 'X-ray crystallography',
            
            # Organisms
            'mouse', 'human', 'zebrafish', 'drosophila', 'C. elegans',
            'E. coli', 'yeast', 'arabidopsis',
            
            # General biology
            'cell culture', 'in vitro', 'in vivo', 'knockout', 'overexpression',
            'pathway', 'signaling', 'regulation', 'expression', 'mutation'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in science_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:15]  # Limit to 15 keywords
    
    def _classify_chunk_type(self, text: str, chunk_index: int) -> str:
        """Simple chunk type classification"""
        text_lower = text.lower()
        
        # First chunk analysis
        if chunk_index == 0:
            if any(word in text_lower for word in ['abstract', 'summary']):
                return 'abstract'
            elif any(word in text_lower for word in ['introduction', 'background']):
                return 'introduction'
            else:
                return 'abstract'  # Default for first chunk
        
        # Content-based classification
        method_indicators = ['method', 'material', 'procedure', 'experimental', 'protocol']
        if any(word in text_lower for word in method_indicators):
            return 'methods'
        
        result_indicators = ['result', 'finding', 'outcome', 'data', 'analysis']
        if any(word in text_lower for word in result_indicators):
            return 'results'
        
        discussion_indicators = ['discussion', 'conclusion', 'implication', 'limitation']
        if any(word in text_lower for word in discussion_indicators):
            return 'discussion'
        
        reference_indicators = ['reference', 'citation', 'bibliography']
        if any(word in text_lower for word in reference_indicators):
            return 'references'
        
        return 'content'  # Default
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF using multiple methods"""
        try:
            filename = Path(pdf_path).name
            logger.info(f"📄 Extracting text from: {filename}")
            
            # Method 1: Try pdfplumber (better for complex layouts)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n\n"
                    
                    if len(full_text.strip()) > 100:
                        logger.info(f"✅ Extracted text using pdfplumber: {len(full_text)} chars")
                        return {
                            'text': full_text.strip(),
                            'method': 'pdfplumber',
                            'page_count': len(pdf.pages),
                            'success': True
                        }
            except Exception as e:
                logger.warning(f"pdfplumber failed for {filename}: {e}")
            
            # Method 2: Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n\n"
                    
                    if len(full_text.strip()) > 100:
                        logger.info(f"✅ Extracted text using PyPDF2: {len(full_text)} chars")
                        return {
                            'text': full_text.strip(),
                            'method': 'PyPDF2',
                            'page_count': len(pdf_reader.pages),
                            'success': True
                        }
            except Exception as e:
                logger.warning(f"PyPDF2 failed for {filename}: {e}")
            
            logger.error(f"❌ All text extraction methods failed for {filename}")
            return {
                'text': '',
                'method': 'failed',
                'page_count': 0,
                'success': False
            }
            
        except Exception as e:
            logger.error(f"Text extraction error for {pdf_path}: {e}")
            return {
                'text': '',
                'method': 'error',
                'page_count': 0,
                'success': False
            }
    
    def create_text_chunks(self, text: str, doc_id: str, title: str) -> List[Dict[str, Any]]:
        """Split text into chunks for embeddings"""
        chunk_size = 8000
        overlap = 200
        
        # Clean text
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = ' '.join(text.split())  # Normalize whitespace
        
        if len(text) <= chunk_size:
            return [{
                'id': f"{doc_id}_chunk_0",
                'text': text,
                'chunk_index': 0,
                'doc_id': doc_id,
                'title': title
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': f"{doc_id}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'doc_id': doc_id,
                    'title': title
                })
                chunk_index += 1
            
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"🧩 Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            logger.info(f"🤖 Generating embeddings for {len(texts)} texts...")
            
            if self.openai_client:
                # New OpenAI client
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002"  # Stable model
                )
                embeddings = [emb.embedding for emb in response.data]
            else:
                # Legacy OpenAI client
                response = openai.Embedding.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )
                embeddings = [emb['embedding'] for emb in response['data']]
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def upload_to_vector_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], 
                             file_metadata: Dict[str, Any], pdf_metadata: Dict[str, Any]) -> bool:
        """Upload chunks and embeddings to Pinecone with enhanced metadata"""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")
            
            logger.info(f"📤 Uploading {len(chunks)} vectors to Pinecone...")
            
            # Prepare vectors with enhanced metadata
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                # Enhanced metadata for each chunk
                enhanced_metadata = {
                    # Original chunk metadata
                    'doc_id': chunk['doc_id'],
                    'title': chunk['title'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'][:1000],  # Truncate for metadata storage
                    'upload_timestamp': datetime.now().isoformat(),
                    
                    # File metadata
                    'filename': file_metadata['filename'],
                    'file_hash': file_metadata['file_hash'],
                    'page_count': file_metadata['page_count'],
                    'word_count': file_metadata['word_count'],
                    'extraction_method': file_metadata['extraction_method'],
                    
                    # Enhanced PDF metadata
                    'doi': pdf_metadata.get('doi'),
                    'journal': pdf_metadata.get('journal'),
                    'publication_year': pdf_metadata.get('publication_year'),
                    'extracted_title': pdf_metadata.get('title'),
                    'abstract': pdf_metadata.get('abstract', '')[:500] if pdf_metadata.get('abstract') else None,
                    
                    # Author information (truncated for metadata limits)
                    'authors': pdf_metadata.get('authors', [])[:5],  # Limit to 5 authors
                    'author_count': len(pdf_metadata.get('authors', [])),
                    
                    # Institution information
                    'institutions': pdf_metadata.get('institutions', [])[:3],  # Limit to 3
                    
                    # Keywords from text analysis
                    'keywords': self._extract_simple_keywords(chunk['text']),
                    
                    # Chunk type classification
                    'chunk_type': self._classify_chunk_type(chunk['text'], chunk['chunk_index']),
                    
                    # Crossref enrichment (if available)
                    'crossref_journal': pdf_metadata.get('crossref_journal'),
                    'crossref_year': pdf_metadata.get('crossref_year'),
                    'citation_count': pdf_metadata.get('citation_count', 0),
                    'publisher': pdf_metadata.get('publisher'),
                    'issn': pdf_metadata.get('issn'),
                    'subject_areas': pdf_metadata.get('subject_areas', [])[:5],  # Limit to 5
                }
                
                # Remove None values to save space
                enhanced_metadata = {k: v for k, v in enhanced_metadata.items() 
                                   if v is not None and v != '' and v != []}
                
                vector_data = {
                    'id': chunk['id'],
                    'values': embedding,
                    'metadata': enhanced_metadata
                }
                vectors.append(vector_data)
            
            # Upload in batches
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    self.pinecone_index.upsert(vectors=batch)
                    logger.info(f"✅ Uploaded batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                except Exception as e:
                    logger.error(f"❌ Batch {batch_num} upload failed: {e}")
                    return False
            
            logger.info(f"🎉 Successfully uploaded all {len(vectors)} vectors with enhanced metadata")
            return True
            
        except Exception as e:
            logger.error(f"Vector store upload failed: {e}")
            return False
    
    def process_single_pdf(self, pdf_path: str) -> bool:
        """Process a single PDF file with enhanced metadata extraction"""
        try:
            filename = Path(pdf_path).name
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Processing: {filename}")
            logger.info(f"{'='*80}")
            
            # Calculate file hash for duplicate detection
            file_hash = self._get_file_hash(pdf_path)
            if not file_hash:
                logger.error(f"❌ Could not calculate hash for {filename}")
                return False
            
            # Check if already processed
            if file_hash in self.processed_files:
                logger.info(f"⏭️ Skipping {filename} (already processed)")
                return True
            
            # Extract text
            extraction_result = self.extract_text_from_pdf(pdf_path)
            if not extraction_result['success']:
                logger.error(f"❌ Text extraction failed for {filename}")
                return False
            
            # Extract PDF metadata
            pdf_metadata = self.extract_pdf_metadata(pdf_path)
            logger.info(f"📋 Extracted metadata: DOI={pdf_metadata.get('doi')}, "
                       f"Authors={len(pdf_metadata.get('authors', []))}, "
                       f"Journal={pdf_metadata.get('journal')}, "
                       f"Year={pdf_metadata.get('publication_year')}")
            
            # Enrich with Crossref if DOI found
            if pdf_metadata.get('doi') and self.enable_crossref:
                crossref_data = self.enrich_with_crossref(pdf_metadata['doi'])
                if crossref_data:
                    # Merge Crossref data - prefer Crossref as authoritative
                    pdf_metadata.update({
                        'authors': crossref_data.get('crossref_authors') or pdf_metadata.get('authors', []),
                        'journal': crossref_data.get('crossref_journal') or pdf_metadata.get('journal'),
                        'publication_year': crossref_data.get('crossref_year') or pdf_metadata.get('publication_year'),
                        'title': crossref_data.get('crossref_title') or pdf_metadata.get('title'),
                        'abstract': crossref_data.get('crossref_abstract') or pdf_metadata.get('abstract'),
                        'citation_count': crossref_data.get('citation_count', 0),
                        'publisher': crossref_data.get('publisher'),
                        'issn': crossref_data.get('issn'),
                        'subject_areas': crossref_data.get('subject_areas', []),
                        'crossref_journal': crossref_data.get('crossref_journal'),
                        'crossref_year': crossref_data.get('crossref_year')
                    })
                    logger.info(f"🌐 Enhanced with Crossref: citations={crossref_data.get('citation_count', 0)}")
            
            # Create document metadata
            doc_id = hashlib.sha256(f"{filename}{file_hash}".encode()).hexdigest()[:16]
            title = pdf_metadata.get('title') or filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            
            # Create chunks
            chunks = self.create_text_chunks(extraction_result['text'], doc_id, title)
            if not chunks:
                logger.error(f"❌ No chunks created for {filename}")
                return False
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Prepare file metadata
            file_metadata = {
                'filename': filename,
                'file_hash': file_hash,
                'doc_id': doc_id,
                'extraction_method': extraction_result['method'],
                'page_count': extraction_result['page_count'],
                'word_count': len(extraction_result['text'].split()),
                'chunk_count': len(chunks)
            }
            
            # Upload to vector store with enhanced metadata
            upload_success = self.upload_to_vector_store(chunks, embeddings, file_metadata, pdf_metadata)
            
            if upload_success:
                # Mark as processed
                self.processed_files[file_hash] = {
                    'filename': filename,
                    'doc_id': doc_id,
                    'processed_at': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'word_count': file_metadata['word_count'],
                    'doi': pdf_metadata.get('doi'),
                    'journal': pdf_metadata.get('journal'),
                    'authors': pdf_metadata.get('authors', []),
                    'publication_year': pdf_metadata.get('publication_year'),
                    'citation_count': pdf_metadata.get('citation_count', 0)
                }
                self._save_processed_files()
                
                logger.info(f"✅ Successfully processed {filename}")
                logger.info(f"   📄 Pages: {extraction_result['page_count']}")
                logger.info(f"   📝 Words: {file_metadata['word_count']}")
                logger.info(f"   🧩 Chunks: {len(chunks)}")
                logger.info(f"   👥 Authors: {len(pdf_metadata.get('authors', []))}")
                logger.info(f"   📰 Journal: {pdf_metadata.get('journal', 'Unknown')}")
                logger.info(f"   📅 Year: {pdf_metadata.get('publication_year', 'Unknown')}")
                logger.info(f"   📊 Citations: {pdf_metadata.get('citation_count', 0)}")
                logger.info(f"   🔗 DOI: {pdf_metadata.get('doi', 'None')}")
                logger.info(f"   🆔 Doc ID: {doc_id}")
                return True
            else:
                logger.error(f"❌ Upload failed for {filename}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_folder(self, folder_path: str) -> Dict[str, int]:
        """Process all PDF files in a folder"""
        try:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # Find all PDF files
            pdf_files = list(folder.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {folder_path}")
                return {'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0}
            
            logger.info(f"🔍 Found {len(pdf_files)} PDF files in {folder_path}")
            
            # Initialize stats
            stats = {
                'total': len(pdf_files),
                'processed': 0,
                'failed': 0,
                'skipped': 0,
                'total_chunks': 0,
                'total_citations': 0,
                'dois_found': 0,
                'crossref_enriched': 0
            }
            
            # Process each file
            for i, pdf_file in enumerate(pdf_files, 1):
                logger.info(f"\n📁 File {i}/{len(pdf_files)}")
                
                # Quick duplicate check
                file_hash = self._get_file_hash(str(pdf_file))
                if file_hash in self.processed_files:
                    stats['skipped'] += 1
                    logger.info(f"⏭️ Skipping {pdf_file.name} (already processed)")
                    continue
                
                # Process the file
                success = self.process_single_pdf(str(pdf_file))
                if success:
                    stats['processed'] += 1
                    # Update additional stats
                    if file_hash in self.processed_files:
                        processed_data = self.processed_files[file_hash]
                        stats['total_chunks'] += processed_data.get('chunk_count', 0)
                        stats['total_citations'] += processed_data.get('citation_count', 0)
                        if processed_data.get('doi'):
                            stats['dois_found'] += 1
                        if processed_data.get('citation_count', 0) > 0:
                            stats['crossref_enriched'] += 1
                else:
                    stats['failed'] += 1
                
                # Save progress periodically
                if i % 5 == 0:
                    self._save_processed_files()
                
                # Small delay to be respectful to APIs
                if self.enable_crossref:
                    time.sleep(0.5)
            
            # Final save
            self._save_processed_files()
            
            # Print comprehensive summary
            logger.info(f"\n{'='*90}")
            logger.info("🎯 ENHANCED PROCESSING SUMMARY")
            logger.info(f"{'='*90}")
            logger.info(f"📁 Folder: {folder_path}")
            logger.info(f"📊 Total files: {stats['total']}")
            logger.info(f"✅ Successfully processed: {stats['processed']}")
            logger.info(f"❌ Failed: {stats['failed']}")
            logger.info(f"⏭️ Skipped (already processed): {stats['skipped']}")
            logger.info(f"🧩 Total chunks created: {stats['total_chunks']}")
            logger.info(f"🔗 DOIs found: {stats['dois_found']}")
            logger.info(f"🌐 Crossref enriched: {stats['crossref_enriched']}")
            logger.info(f"📊 Total citations tracked: {stats['total_citations']}")
            
            if stats['total'] - stats['skipped'] > 0:
                success_rate = (stats['processed'] / (stats['total'] - stats['skipped'])) * 100
                logger.info(f"📈 Success rate: {success_rate:.1f}%")
            
            logger.info(f"{'='*90}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Folder processing failed: {e}")
            return {'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0, 'error': str(e)}
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processed files"""
        try:
            if not self.processed_files:
                return {'total_files': 0, 'message': 'No files processed yet'}
            
            # Analyze processed files
            total_files = len(self.processed_files)
            total_chunks = sum(f.get('chunk_count', 0) for f in self.processed_files.values())
            total_words = sum(f.get('word_count', 0) for f in self.processed_files.values())
            total_citations = sum(f.get('citation_count', 0) for f in self.processed_files.values())
            
            # Count files with metadata
            dois_count = sum(1 for f in self.processed_files.values() if f.get('doi'))
            journals_count = sum(1 for f in self.processed_files.values() if f.get('journal'))
            authors_count = sum(1 for f in self.processed_files.values() if f.get('authors'))
            
            # Get year distribution
            years = [f.get('publication_year') for f in self.processed_files.values() if f.get('publication_year')]
            year_counts = {}
            for year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
            
            # Get journal distribution
            journals = [f.get('journal') for f in self.processed_files.values() if f.get('journal')]
            journal_counts = {}
            for journal in journals:
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
            
            # Top journals
            top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            summary = {
                'total_files': total_files,
                'total_chunks': total_chunks,
                'total_words': total_words,
                'total_citations': total_citations,
                'files_with_dois': dois_count,
                'files_with_journals': journals_count,
                'files_with_authors': authors_count,
                'year_distribution': dict(sorted(year_counts.items(), reverse=True)[:10]),
                'top_journals': top_journals,
                'metadata_coverage': {
                    'dois': f"{(dois_count/total_files)*100:.1f}%",
                    'journals': f"{(journals_count/total_files)*100:.1f}%",
                    'authors': f"{(authors_count/total_files)*100:.1f}%"
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate processing summary: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    
    logger.info("🚀 Starting Enhanced PDF Processing Pipeline")
    logger.info("=" * 80)
    
    # Parse command line arguments
    if len(sys.argv) < 5:
        logger.error("❌ Missing arguments")
        print("\nUsage: python enhanced_pdf_pipeline.py <folder_path> <openai_key> <pinecone_key> <index_name>")
        print("\nExample:")
        print("python enhanced_pdf_pipeline.py /path/to/pdfs sk-abc123... xyz-pinecone-key... genomics-publications")
        print("\nEnvironment variables (optional):")
        print("  EXTRACT_PDF_METADATA=true/false (default: true)")
        print("  ENRICH_WITH_CROSSREF=true/false (default: true)")
        print("  CROSSREF_EMAIL=your-email@domain.com")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    openai_key = sys.argv[2]
    pinecone_key = sys.argv[3]
    index_name = sys.argv[4]
    
    # Validate inputs
    if not Path(folder_path).exists():
        logger.error(f"❌ Folder does not exist: {folder_path}")
        sys.exit(1)
    
    if not openai_key.startswith('sk-'):
        logger.warning("⚠️ OpenAI key doesn't start with 'sk-' - this might be incorrect")
    
    logger.info(f"📁 Processing folder: {folder_path}")
    logger.info(f"🎯 Target index: {index_name}")
    logger.info(f"🔬 Metadata extraction: {os.getenv('EXTRACT_PDF_METADATA', 'true')}")
    logger.info(f"🌐 Crossref enrichment: {os.getenv('ENRICH_WITH_CROSSREF', 'true')}")
    
    try:
        # Initialize enhanced pipeline
        logger.info("🔧 Initializing enhanced pipeline...")
        pipeline = EnhancedPDFPipeline(openai_key, pinecone_key, index_name)
        
        # Process all PDFs
        stats = pipeline.process_folder(folder_path)
        
        # Show processing summary
        logger.info("\n📋 Generating processing summary...")
        summary = pipeline.get_processing_summary()
        
        if 'error' not in summary:
            logger.info(f"\n📊 PROCESSING SUMMARY:")
            logger.info(f"   Files processed: {summary['total_files']}")
            logger.info(f"   Total chunks: {summary['total_chunks']}")
            logger.info(f"   Total words: {summary['total_words']:,}")
            logger.info(f"   Total citations: {summary['total_citations']}")
            logger.info(f"   Metadata coverage:")
            logger.info(f"     DOIs: {summary['metadata_coverage']['dois']}")
            logger.info(f"     Journals: {summary['metadata_coverage']['journals']}")
            logger.info(f"     Authors: {summary['metadata_coverage']['authors']}")
            
            if summary['top_journals']:
                logger.info(f"   Top journals:")
                for journal, count in summary['top_journals'][:5]:
                    logger.info(f"     {journal}: {count} papers")
        
        # Exit with appropriate code
        if 'error' in stats:
            logger.error(f"❌ Pipeline failed: {stats['error']}")
            sys.exit(1)
        elif stats['failed'] > 0:
            logger.warning(f"⚠️ Some files failed: {stats['failed']} failures")
            sys.exit(2)
        else:
            logger.info("🎉 All files processed successfully!")
            logger.info("\n💡 Next steps:")
            logger.info("   - Test enhanced search with filters")
            logger.info("   - Query by author, journal, or year")
            logger.info("   - Analyze your research collection")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```
```
#scripts/analytics.py
#!/usr/bin/env python3
"""
Analytics script for enhanced vector store
"""
import sys
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pinecone import Pinecone

def get_vector_store_analytics():
    """Get comprehensive analytics from vector store"""
    try:
        # Initialize Pinecone
        api_key =""

        index_name = 'genomics-publications'
        
        if not api_key:
            print("❌ PINECONE_API_KEY environment variable required")
            return
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        print("📊 Vector Store Analytics")
        print("=" * 60)
        
        # Get basic index stats
        stats = index.describe_index_stats()
        print(f"Total vectors: {stats.total_vector_count:,}")
        print(f"Dimension: {stats.dimension}")
        print(f"Index fullness: {stats.index_fullness:.2%}")
        
        # Sample vectors to analyze metadata
        print("\n🔍 Analyzing metadata from sample vectors...")
        
        # Query with dummy vector to get sample results
        dummy_vector = [0.0] * stats.dimension
        
        # Get samples from different namespaces if they exist
        sample_results = index.query(
            vector=dummy_vector,
            top_k=1000,  # Get larger sample
            include_metadata=True
        )
        
        if not sample_results.matches:
            print("⚠️ No vectors found in index")
            return
        
        print(f"Analyzing {len(sample_results.matches)} sample vectors...")
        
        # Analyze metadata
        metadata_analysis = analyze_metadata([match.metadata for match in sample_results.matches])
        
        # Print results
        print_metadata_analysis(metadata_analysis)
        
        # Load processed files summary if available
        processed_file = 'processed_files.json'
        if Path(processed_file).exists():
            print(f"\n📋 Processing History Analysis")
            print("-" * 40)
            with open(processed_file, 'r') as f:
                processed_data = json.load(f)
            
            print_processing_history(processed_data)
        
    except Exception as e:
        print(f"❌ Analytics failed: {e}")

def analyze_metadata(metadata_list):
    """Analyze metadata from vector samples"""
    analysis = {
        'total_samples': len(metadata_list),
        'journals': Counter(),
        'years': Counter(),
        'authors': Counter(),
        'institutions': Counter(),
        'chunk_types': Counter(),
        'keywords': Counter(),
        'publishers': Counter(),
        'dois_found': 0,
        'citations': [],
        'coverage': {}
    }
    
    for metadata in metadata_list:
        if not metadata:
            continue
        
        # Journal analysis
        if metadata.get('journal'):
            analysis['journals'][metadata['journal']] += 1
        if metadata.get('crossref_journal'):
            analysis['journals'][metadata['crossref_journal']] += 1
        
        # Year analysis
        if metadata.get('publication_year'):
            analysis['years'][metadata['publication_year']] += 1
        if metadata.get('crossref_year'):
            analysis['years'][metadata['crossref_year']] += 1
        
        # Author analysis
        authors = metadata.get('authors', [])
        if isinstance(authors, list):
            for author in authors[:3]:  # Top 3 authors per paper
                if author and len(str(author)) > 3:
                    analysis['authors'][str(author)] += 1
        
        # Institution analysis
        institutions = metadata.get('institutions', [])
        if isinstance(institutions, list):
            for inst in institutions:
                if inst and len(str(inst)) > 5:
                    analysis['institutions'][str(inst)] += 1
        
        # Chunk type analysis
        if metadata.get('chunk_type'):
            analysis['chunk_types'][metadata['chunk_type']] += 1
        
        # Keywords analysis
        keywords = metadata.get('keywords', [])
        if isinstance(keywords, list):
            for keyword in keywords:
                if keyword:
                    analysis['keywords'][str(keyword)] += 1
        
        # Publisher analysis
        if metadata.get('publisher'):
            analysis['publishers'][metadata['publisher']] += 1
        
        # DOI coverage
        if metadata.get('doi'):
            analysis['dois_found'] += 1
        
        # Citation analysis
        if metadata.get('citation_count'):
            try:
                citations = int(metadata['citation_count'])
                analysis['citations'].append(citations)
            except:
                pass
    
    # Calculate coverage percentages
    total = analysis['total_samples']
    analysis['coverage'] = {
        'journals': (len([m for m in metadata_list if m.get('journal') or m.get('crossref_journal')]) / total) * 100,
        'years': (len([m for m in metadata_list if m.get('publication_year') or m.get('crossref_year')]) / total) * 100,
        'authors': (len([m for m in metadata_list if m.get('authors')]) / total) * 100,
        'dois': (analysis['dois_found'] / total) * 100,
        'institutions': (len([m for m in metadata_list if m.get('institutions')]) / total) * 100,
        'keywords': (len([m for m in metadata_list if m.get('keywords')]) / total) * 100
    }
    
    return analysis

def print_metadata_analysis(analysis):
    """Print formatted metadata analysis"""
    print(f"\n📈 Metadata Coverage Analysis")
    print("-" * 40)
    print(f"Sample size: {analysis['total_samples']:,} vectors")
    print(f"Journal coverage: {analysis['coverage']['journals']:.1f}%")
    print(f"Year coverage: {analysis['coverage']['years']:.1f}%")
    print(f"Author coverage: {analysis['coverage']['authors']:.1f}%")
    print(f"DOI coverage: {analysis['coverage']['dois']:.1f}%")
    print(f"Institution coverage: {analysis['coverage']['institutions']:.1f}%")
    print(f"Keywords coverage: {analysis['coverage']['keywords']:.1f}%")
    
    # Top journals
    if analysis['journals']:
        print(f"\n📰 Top Journals ({len(analysis['journals'])} total)")
        print("-" * 40)
        for journal, count in analysis['journals'].most_common(10):
            percentage = (count / analysis['total_samples']) * 100
            print(f"{journal[:50]:<50} {count:>4} ({percentage:.1f}%)")
    
    # Publication years
    if analysis['years']:
        print(f"\n📅 Publication Years")
        print("-" * 40)
        years_sorted = sorted(analysis['years'].items(), key=lambda x: x[0], reverse=True)
        for year, count in years_sorted[:10]:
            percentage = (count / analysis['total_samples']) * 100
            print(f"{year:<10} {count:>4} papers ({percentage:.1f}%)")
    
    # Top authors
    if analysis['authors']:
        print(f"\n👥 Most Frequent Authors")
        print("-" * 40)
        for author, count in analysis['authors'].most_common(10):
            print(f"{author[:40]:<40} {count:>4} papers")
    
    # Top institutions
    if analysis['institutions']:
        print(f"\n🏛️ Top Institutions")
        print("-" * 40)
        for inst, count in analysis['institutions'].most_common(10):
            print(f"{inst[:45]:<45} {count:>4} papers")
    
    # Chunk types
    if analysis['chunk_types']:
        print(f"\n📄 Chunk Type Distribution")
        print("-" * 40)
        for chunk_type, count in analysis['chunk_types'].most_common():
            percentage = (count / analysis['total_samples']) * 100
            print(f"{chunk_type:<15} {count:>6} ({percentage:.1f}%)")
    
    # Top keywords
    if analysis['keywords']:
        print(f"\n🔤 Most Common Keywords")
        print("-" * 40)
        for keyword, count in analysis['keywords'].most_common(15):
            print(f"{keyword:<25} {count:>4}")
    
    # Citation statistics
    if analysis['citations']:
        citations = analysis['citations']
        print(f"\n📊 Citation Statistics")
        print("-" * 40)
        print(f"Papers with citations: {len(citations)}")
        print(f"Total citations: {sum(citations):,}")
        print(f"Average citations: {np.mean(citations):.1f}")
        print(f"Median citations: {np.median(citations):.1f}")
        print(f"Max citations: {max(citations):,}")
        print(f"High-impact papers (>100 citations): {len([c for c in citations if c > 100])}")

def print_processing_history(processed_data):
    """Print processing history analysis"""
    if not processed_data:
        print("No processing history found")
        return
    
    files = list(processed_data.values())
    
    print(f"Total files processed: {len(files)}")
    
    # Processing dates
    dates = []
    for file_data in files:
        if file_data.get('processed_at'):
            try:
                date = datetime.fromisoformat(file_data['processed_at'].replace('Z', '+00:00'))
                dates.append(date)
            except:
                pass
    
    if dates:
        dates.sort()
        print(f"Processing period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # File statistics
    total_chunks = sum(f.get('chunk_count', 0) for f in files)
    total_words = sum(f.get('word_count', 0) for f in files)
    total_citations = sum(f.get('citation_count', 0) for f in files)
    
    print(f"Total chunks: {total_chunks:,}")
    print(f"Total words: {total_words:,}")
    print(f"Total citations: {total_citations:,}")
    
    # DOI coverage
    dois_count = sum(1 for f in files if f.get('doi'))
    print(f"Files with DOIs: {dois_count}/{len(files)} ({(dois_count/len(files))*100:.1f}%)")
    
    # Journal distribution
    journals = [f.get('journal') for f in files if f.get('journal')]
    if journals:
        journal_counts = Counter(journals)
        print(f"\nTop journals in collection:")
        for journal, count in journal_counts.most_common(5):
            print(f"  {journal}: {count} papers")

def export_analytics_report():
    """Export detailed analytics to JSON file"""
    try:
        # Get analytics data
        api_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        if not api_key:
            print("❌ PINECONE_API_KEY required for export")
            return
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Get sample data
        stats = index.describe_index_stats()
        dummy_vector = [0.0] * stats.dimension
        sample_results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True
        )
        
        # Analyze
        metadata_analysis = analyze_metadata([match.metadata for match in sample_results.matches])
        
        # Prepare report
        report = {
            'generated_at': datetime.now().isoformat(),
            'index_stats': {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            },
            'metadata_analysis': {
                'sample_size': metadata_analysis['total_samples'],
                'coverage': metadata_analysis['coverage'],
                'top_journals': dict(metadata_analysis['journals'].most_common(20)),
                'year_distribution': dict(metadata_analysis['years']),
                'top_authors': dict(metadata_analysis['authors'].most_common(50)),
                'top_institutions': dict(metadata_analysis['institutions'].most_common(30)),
                'chunk_types': dict(metadata_analysis['chunk_types']),
                'top_keywords': dict(metadata_analysis['keywords'].most_common(50)),
                'citation_stats': {
                    'total_citations': sum(metadata_analysis['citations']),
                    'avg_citations': np.mean(metadata_analysis['citations']) if metadata_analysis['citations'] else 0,
                    'max_citations': max(metadata_analysis['citations']) if metadata_analysis['citations'] else 0
                }
            }
        }
        
        # Save report
        report_file = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report exported to: {report_file}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze enhanced vector store')
    parser.add_argument('--export', action='store_true', help='Export detailed report to JSON')
    
    args = parser.parse_args()
    
    if args.export:
        export_analytics_report()
    else:
        get_vector_store_analytics()

if __name__ == "__main__":
    main()
```
```
#scripts/test_enhanced_search.py
#!/usr/bin/env python3
"""
Test enhanced search capabilities of the vector store
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pinecone import Pinecone
import openai

class EnhancedVectorSearch:
    """Enhanced search functions for vector store"""
    
    def __init__(self, pinecone_api_key: str = None, openai_api_key: str = None, index_name: str = None):
        self.pinecone_api_key = pinecone_api_key or os.getenv('')
        self.openai_api_key = openai_api_key or os.getenv('')
        self.index_name = index_name or os.getenv('genomics-publications')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        # Initialize clients
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        self.openai_client = self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            # Test connection
            client.models.list()
            return client
        except:
            # Legacy fallback
            openai.api_key = self.openai_api_key
            return None
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for query text"""
        try:
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    input=[query_text],
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            else:
                response = openai.Embedding.create(
                    input=[query_text],
                    model="text-embedding-ada-002"
                )
                return response['data'][0]['embedding']
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            return None
    
    def search_with_filters(self, 
                          query_vector: List[float],
                          author: str = None,
                          journal: str = None,
                          year_range: tuple = None,
                          keywords: List[str] = None,
                          institution: str = None,
                          chunk_type: str = None,
                          min_citations: int = None,
                          publisher: str = None,
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search with multiple filters"""
        
        # Build metadata filter
        filter_conditions = {}
        
        if author:
            filter_conditions["authors"] = {"$in": [author]}
        
        if journal:
            # Search in both journal fields
            filter_conditions["$or"] = [
                {"journal": {"$eq": journal}},
                {"crossref_journal": {"$eq": journal}}
            ]
        
        if year_range:
            start_year, end_year = year_range
            # Search in both year fields
            year_filter = {
                "$or": [
                    {"publication_year": {"$gte": start_year, "$lte": end_year}},
                    {"crossref_year": {"$gte": start_year, "$lte": end_year}}
                ]
            }
            if filter_conditions.get("$or"):
                filter_conditions["$and"] = [
                    {"$or": filter_conditions.pop("$or")},
                    year_filter
                ]
            else:
                filter_conditions.update(year_filter)
        
        if keywords:
            filter_conditions["keywords"] = {"$in": keywords}
        
        if institution:
            filter_conditions["institutions"] = {"$in": [institution]}
        
        if chunk_type:
            filter_conditions["chunk_type"] = {"$eq": chunk_type}
        
        if min_citations:
            filter_conditions["citation_count"] = {"$gte": min_citations}
        
        if publisher:
            filter_conditions["publisher"] = {"$eq": publisher}
        
        # Perform search
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_conditions if filter_conditions else None,
                include_metadata=True,
                include_values=False
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def search_by_text(self, query_text: str, **kwargs) -> List[Dict[str, Any]]:
        """Search using text query (generates embedding automatically)"""
        query_vector = self.generate_query_embedding(query_text)
        if not query_vector:
            return []
        
        return self.search_with_filters(query_vector=query_vector, **kwargs)
    
    def find_similar_papers_by_author(self, author_name: str, query_text: str, top_k: int = 5):
        """Find papers similar to query by specific author"""
        return self.search_by_text(
            query_text=query_text,
            author=author_name,
            top_k=top_k
        )
    
    def get_papers_by_year(self, year: int, top_k: int = 20):
        """Get papers from a specific year"""
        # Use a neutral query for year-based search
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            year_range=(year, year),
            top_k=top_k
        )
    
    def get_high_impact_papers(self, min_citations: int = 50, top_k: int = 10):
        """Get highly cited papers"""
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            min_citations=min_citations,
            top_k=top_k
        )
    
    def search_abstracts_only(self, query_text: str, top_k: int = 10):
        """Search only in abstract chunks"""
        return self.search_by_text(
            query_text=query_text,
            chunk_type="abstract",
            top_k=top_k
        )
    
    def get_papers_by_institution(self, institution: str, top_k: int = 15):
        """Get papers from specific institution"""
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            institution=institution,
            top_k=top_k
        )

def print_search_results(results: List[Dict[str, Any]], title: str):
    """Print formatted search results"""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Title: {metadata.get('title', 'N/A')[:80]}...")
        print(f"   Journal: {metadata.get('journal') or metadata.get('crossref_journal', 'N/A')}")
        print(f"   Year: {metadata.get('publication_year') or metadata.get('crossref_year', 'N/A')}")
        print(f"   Authors: {', '.join(metadata.get('authors', [])[:3])}")
        print(f"   Citations: {metadata.get('citation_count', 0)}")
        if metadata.get('doi'):
            print(f"   DOI: {metadata['doi']}")
        print(f"   Chunk: {metadata.get('chunk_type', 'N/A')} (index: {metadata.get('chunk_index', 'N/A')})")

def test_basic_search(searcher: EnhancedVectorSearch):
    """Test basic text search"""
    print("🔍 Testing Basic Text Search")
    
    query = "CRISPR gene editing"
    results = searcher.search_by_text(query, top_k=5)
    print_search_results(results, f"Results for: '{query}'")

def test_filtered_searches(searcher: EnhancedVectorSearch):
    """Test various filtered searches"""
    
    # Test journal filter
    print("\n📰 Testing Journal Filter")
    results = searcher.search_by_text(
        query_text="machine learning",
        journal="Nature",
        top_k=3
    )
    print_search_results(results, "Machine learning papers in Nature")
    
    # Test year range filter
    print("\n📅 Testing Year Filter")
    results = searcher.search_by_text(
        query_text="RNA sequencing",
        year_range=(2020, 2024),
        top_k=3
    )
    print_search_results(results, "RNA sequencing papers (2020-2024)")
    
    # Test chunk type filter (abstracts only)
    print("\n📄 Testing Chunk Type Filter")
    results = searcher.search_abstracts_only("single cell analysis", top_k=3)
    print_search_results(results, "Single cell analysis (abstracts only)")
    
    # Test high-impact papers
    print("\n📊 Testing Citation Filter")
    results = searcher.get_high_impact_papers(min_citations=20, top_k=3)
    print_search_results(results, "High-impact papers (20+ citations)")

def test_specialized_searches(searcher: EnhancedVectorSearch):
    """Test specialized search functions"""
    
    # Test author search
    print("\n👥 Testing Author Search")
    # Get some authors from recent results first
    sample_results = searcher.search_by_text("genomics", top_k=10)
    
    authors_found = []
    for result in sample_results:
        authors = result['metadata'].get('authors', [])
        if authors:
            authors_found.extend(authors[:2])  # Take first 2 authors
    
    if authors_found:
        test_author = authors_found[0]
        results = searcher.find_similar_papers_by_author(
            author_name=test_author,
            query_text="research",
            top_k=3
        )
        print_search_results(results, f"Papers by author: {test_author}")
    else:
        print("No authors found in sample data")
    
    # Test institution search
    print("\n🏛️ Testing Institution Search")
    # Get some institutions from recent results
    institutions_found = []
    for result in sample_results:
        institutions = result['metadata'].get('institutions', [])
        if institutions:
            institutions_found.extend(institutions[:2])
    
    if institutions_found:
        test_institution = institutions_found[0]
        results = searcher.get_papers_by_institution(test_institution, top_k=3)
        print_search_results(results, f"Papers from: {test_institution}")
    else:
        print("No institutions found in sample data")

def test_keyword_search(searcher: EnhancedVectorSearch):
    """Test keyword-based filtering"""
    print("\n🔤 Testing Keyword Search")
    
    # Test with common scientific keywords
    keywords_to_test = ["CRISPR", "RNA", "protein", "cancer", "genomics"]
    
    for keyword in keywords_to_test:
        results = searcher.search_by_text(
            query_text="biological research",
            keywords=[keyword],
            top_k=2
        )
        
        if results:
            print(f"\nPapers with keyword '{keyword}':")
            for result in results[:2]:
                metadata = result['metadata']
                title = metadata.get('title', 'N/A')[:60]
                journal = metadata.get('journal') or metadata.get('crossref_journal', 'N/A')
                print(f"  • {title}... ({journal})")
            break
    else:
        print("No results found for tested keywords")

def run_search_demo():
    """Run comprehensive search demonstration"""
    try:
        print("🚀 Enhanced Vector Search Test Suite")
        print("=" * 50)
        
        # Initialize searcher
        searcher = EnhancedVectorSearch()
        
        # Get index info
        stats = searcher.index.describe_index_stats()
        print(f"Index: {searcher.index_name}")
        print(f"Total vectors: {stats.total_vector_count:,}")
        print(f"Dimension: {stats.dimension}")
        
        # Run tests
        test_basic_search(searcher)
        test_filtered_searches(searcher)
        test_specialized_searches(searcher)
        test_keyword_search(searcher)
        
        print(f"\n✅ Search testing completed!")
        print("\n💡 Try these search patterns in your application:")
        print("   - searcher.search_by_text('your query')")
        print("   - searcher.search_by_text('query', journal='Nature', year_range=(2020, 2024))")
        print("   - searcher.search_abstracts_only('your query')")
        print("   - searcher.get_high_impact_papers(min_citations=50)")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()

def interactive_search():
    """Interactive search mode"""
    try:
        searcher = EnhancedVectorSearch()
        
        print("🔍 Interactive Enhanced Search")
        print("=" * 40)
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Ask for filters
            print("\nOptional filters (press Enter to skip):")
            journal = input("Journal name: ").strip() or None
            author = input("Author name: ").strip() or None
            year_input = input("Year (YYYY) or range (YYYY-YYYY): ").strip()
            chunk_type = input("Chunk type (abstract/methods/results/discussion): ").strip() or None
            
            # Parse year input
            year_range = None
            if year_input:
                if '-' in year_input:
                    try:
                        start, end = year_input.split('-')
                        year_range = (int(start), int(end))
                    except:
                        print("Invalid year range format")
                else:
                    try:
                        year = int(year_input)
                        year_range = (year, year)
                    except:
                        print("Invalid year format")
            
            # Perform search
            print(f"\nSearching for: '{query}'...")
            results = searcher.search_by_text(
                query_text=query,
                journal=journal,
                author=author,
                year_range=year_range,
                chunk_type=chunk_type,
                top_k=5
            )
            
            print_search_results(results, f"Search Results")
    
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user")
    except Exception as e:
        print(f"❌ Interactive search failed: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced search capabilities')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run interactive search mode')
    parser.add_argument('--query', '-q', type=str, help='Run single query test')
    parser.add_argument('--journal', type=str, help='Filter by journal')
    parser.add_argument('--author', type=str, help='Filter by author')
    parser.add_argument('--year', type=str, help='Filter by year (YYYY or YYYY-YYYY)')
    parser.add_argument('--citations', type=int, help='Minimum citation count')
    parser.add_argument('--chunk-type', type=str, choices=['abstract', 'methods', 'results', 'discussion', 'content'], help='Filter by chunk type')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_search()
    elif args.query:
        # Single query mode
        try:
            searcher = EnhancedVectorSearch()
            
            # Parse year range
            year_range = None
            if args.year:
                if '-' in args.year:
                    start, end = args.year.split('-')
                    year_range = (int(start), int(end))
                else:
                    year = int(args.year)
                    year_range = (year, year)
            
            print(f"🔍 Searching for: '{args.query}'")
            if args.journal:
                print(f"   Journal filter: {args.journal}")
            if args.author:
                print(f"   Author filter: {args.author}")
            if year_range:
                print(f"   Year filter: {year_range[0]}-{year_range[1]}")
            if args.citations:
                print(f"   Min citations: {args.citations}")
            if args.chunk_type:
                print(f"   Chunk type: {args.chunk_type}")
            
            results = searcher.search_by_text(
                query_text=args.query,
                journal=args.journal,
                author=args.author,
                year_range=year_range,
                min_citations=args.citations,
                chunk_type=args.chunk_type,
                top_k=args.top_k
            )
            
            print_search_results(results, f"Search Results ({len(results)} found)")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
    else:
        # Default: run demo
        run_search_demo()

if __name__ == "__main__":
    main()
```
```
#enhanced_search.py 
#!/usr/bin/env python3
"""
Enhanced search functions for your vector store
"""
import os
import openai
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedVectorSearch:
    """Enhanced search functions for vector store"""
    
    def __init__(self, pinecone_api_key: str = None, openai_api_key: str = None, index_name: str = None):
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        # Initialize clients
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        self.openai_client = self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            # Test connection
            client.models.list()
            return client
        except:
            # Legacy fallback
            openai.api_key = self.openai_api_key
            return None
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for query text"""
        try:
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    input=[query_text],
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            else:
                response = openai.Embedding.create(
                    input=[query_text],
                    model="text-embedding-ada-002"
                )
                return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def search_with_filters(self, 
                          query_vector: List[float],
                          author: str = None,
                          journal: str = None,
                          year_range: tuple = None,
                          keywords: List[str] = None,
                          institution: str = None,
                          chunk_type: str = None,
                          min_citations: int = None,
                          publisher: str = None,
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search with multiple filters"""
        
        # Build metadata filter
        filter_conditions = {}
        
        if author:
            filter_conditions["authors"] = {"$in": [author]}
        
        if journal:
            # Search in both journal fields
            filter_conditions["$or"] = [
                {"journal": {"$eq": journal}},
                {"crossref_journal": {"$eq": journal}}
            ]
        
        if year_range:
            start_year, end_year = year_range
            # Search in both year fields
            year_filter = {
                "$or": [
                    {"publication_year": {"$gte": start_year, "$lte": end_year}},
                    {"crossref_year": {"$gte": start_year, "$lte": end_year}}
                ]
            }
            if filter_conditions.get("$or"):
                filter_conditions["$and"] = [
                    {"$or": filter_conditions.pop("$or")},
                    year_filter
                ]
            else:
                filter_conditions.update(year_filter)
        
        if keywords:
            filter_conditions["keywords"] = {"$in": keywords}
        
        if institution:
            filter_conditions["institutions"] = {"$in": [institution]}
        
        if chunk_type:
            filter_conditions["chunk_type"] = {"$eq": chunk_type}
        
        if min_citations:
            filter_conditions["citation_count"] = {"$gte": min_citations}
        
        if publisher:
            filter_conditions["publisher"] = {"$eq": publisher}
        
        # Perform search
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_conditions if filter_conditions else None,
                include_metadata=True,
                include_values=False
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text(self, query_text: str, **kwargs) -> List[Dict[str, Any]]:
        """Search using text query (generates embedding automatically)"""
        query_vector = self.generate_query_embedding(query_text)
        if not query_vector:
            return []
        
        return self.search_with_filters(query_vector=query_vector, **kwargs)
    
    def find_similar_papers_by_author(self, author_name: str, query_text: str, top_k: int = 5):
        """Find papers similar to query by specific author"""
        return self.search_by_text(
            query_text=query_text,
            author=author_name,
            top_k=top_k
        )
    
    def get_papers_by_year(self, year: int, top_k: int = 20):
        """Get papers from a specific year"""
        # Use a neutral query for year-based search
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            year_range=(year, year),
            top_k=top_k
        )
    
    def get_high_impact_papers(self, min_citations: int = 50, top_k: int = 10):
        """Get highly cited papers"""
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            min_citations=min_citations,
            top_k=top_k
        )
    
    def search_abstracts_only(self, query_text: str, top_k: int = 10):
        """Search only in abstract chunks"""
        return self.search_by_text(
            query_text=query_text,
            chunk_type="abstract",
            top_k=top_k
        )
    
    def get_papers_by_institution(self, institution: str, top_k: int = 15):
        """Get papers from specific institution"""
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            institution=institution,
            top_k=top_k
        )
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics from your vector store"""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.total_vector_count,
                "namespace_counts": stats.namespaces,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
            
        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            return {}
```
```
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
```
```
#scripts/migrate_existing_data.py
#!/usr/bin/env python3
"""
Migrate existing Pinecone vectors to enhanced metadata format
"""
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pinecone import Pinecone
from enhanced_pdf_pipeline import EnhancedPDFPipeline

def migrate_existing_vectors():
    """Migrate existing vectors to enhanced metadata format"""
    try:
        print("🔄 Migrating Existing Vectors to Enhanced Format")
        print("=" * 60)
        
        # Get API keys
        pinecone_key = os.getenv('PINECONE_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        if not pinecone_key or not openai_key:
            print("❌ PINECONE_API_KEY and OPENAI_API_KEY required")
            return False
        
        # Initialize clients
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index: {index_name}")
        print(f"Total vectors: {stats.total_vector_count:,}")
        
        # Sample existing vectors to check format
        print("\n🔍 Analyzing existing vector format...")
        dummy_vector = [0.0] * stats.dimension
        sample_results = index.query(
            vector=dummy_vector,
            top_k=10,
            include_metadata=True
        )
        
        if not sample_results.matches:
            print("⚠️ No vectors found in index")
            return False
        
        # Analyze existing metadata format
        old_format_count = 0
        enhanced_format_count = 0
        
        for match in sample_results.matches:
            metadata = match.metadata or {}
            
            # Check if already in enhanced format
            if metadata.get('crossref_journal') or metadata.get('citation_count'):
                enhanced_format_count += 1
            else:
                old_format_count += 1
        
        print(f"Enhanced format vectors: {enhanced_format_count}")
        print(f"Old format vectors: {old_format_count}")
        
        if old_format_count == 0:
            print("✅ All vectors already in enhanced format!")
            return True
        
        # Check if we have original PDFs to reprocess
        processed_files_path = 'processed_files.json'
        if not Path(processed_files_path).exists():
            print("❌ processed_files.json not found - cannot reprocess PDFs")
            print("💡 To migrate existing data, you need to:")
            print("   1. Have original PDF files")
            print("   2. Reprocess them with enhanced pipeline")
            return False
        
        with open(processed_files_path, 'r') as f:
            processed_files = json.load(f)
        
        print(f"\n📁 Found {len(processed_files)} processed files")
        
        # Ask user for confirmation
        response = input("\n❓ Do you want to reprocess PDFs with enhanced metadata? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled by user")
            return False
        
        # Find PDF files
        pdf_folders = []
        for file_data in processed_files.values():
            filename = file_data.get('filename', '')
            if filename.endswith('.pdf'):
                # Common PDF folder locations
                possible_paths = [
                    f"./pdfs/{filename}",
                    f"./data/{filename}",
                    f"./documents/{filename}",
                    f"./{filename}"
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        folder = str(Path(path).parent)
                        if folder not in pdf_folders:
                            pdf_folders.append(folder)
                        break
        
        if not pdf_folders:
            print("❌ No PDF files found in common locations")
            print("💡 Please specify PDF folder location:")
            pdf_folder = input("PDF folder path: ").strip()
            
            if not pdf_folder or not Path(pdf_folder).exists():
                print("Invalid folder path")
                return False
            
            pdf_folders = [pdf_folder]
        
        # Initialize enhanced pipeline
        print("\n🔧 Initializing enhanced pipeline...")
        pipeline = EnhancedPDFPipeline(openai_key, pinecone_key, index_name)
        
        # Clear processed files to force reprocessing
        backup_processed = processed_files.copy()
        pipeline.processed_files = {}
        
        # Process each PDF folder
        total_processed = 0
        for folder in pdf_folders:
            print(f"\n📁 Processing folder: {folder}")
            stats = pipeline.process_folder(folder)
            total_processed += stats.get('processed', 0)
        
        # Restore processed files with new data
        pipeline.processed_files.update(backup_processed)
        pipeline._save_processed_files()
        
        print(f"\n✅ Migration completed!")
        print(f"   Reprocessed: {total_processed} files")
        print(f"   Enhanced metadata added to vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_metadata_enhancement():
    """Analyze the enhancement of metadata in vectors"""
    try:
        print("\n📊 Analyzing Metadata Enhancement")
        print("-" * 40)
        
        pinecone_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        
        # Get larger sample
        dummy_vector = [0.0] * 1536  # Adjust if different dimension
        sample_results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True
        )
        
        enhanced_features = {
            'doi': 0,
            'crossref_journal': 0,
            'citation_count': 0,
            'authors': 0,
            'institutions': 0,
            'keywords': 0,
            'chunk_type': 0
        }
        
        total_vectors = len(sample_results.matches)
        
        for match in sample_results.matches:
            metadata = match.metadata or {}
            
            for feature in enhanced_features:
                if metadata.get(feature):
                    enhanced_features[feature] += 1
        
        print(f"Sample size: {total_vectors} vectors")
        print("\nEnhanced metadata coverage:")
        for feature, count in enhanced_features.items():
            percentage = (count / total_vectors) * 100
            print(f"  {feature:<20} {count:>3}/{total_vectors} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate existing vectors to enhanced format')
    parser.add_argument('--analyze', action='store_true', help='Analyze current metadata enhancement')
    parser.add_argument('--force', action='store_true', help='Force migration without confirmation')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_metadata_enhancement()
    else:
        success = migrate_existing_vectors()
        if success:
            analyze_metadata_enhancement()

if __name__ == "__main__":
    main()
```
```
# 2. Install dependencies
pip install PyPDF2==3.0.1 pdfplumber==0.10.3 pinecone-client==2.2.4 openai==1.3.5 python-dotenv==1.0.0

# 3. Create .env file with your keys
cat > .env << 'EOF'
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications
EOF

# 4. Run ingestion on your PDF folder
python pdf_ingestion_pipeline.py /path/to/your/pdf/folder \
  --openai-key "your_openai_key" \
  --pinecone-key "your_pinecone_key" \
  --index-name "genomics-publications"
```

## 📊 **Example Output:**

```
2025-07-15 05:21:58,112 - INFO - 🚀 Starting Enhanced PDF Processing Pipeline
2025-07-15 05:21:58,112 - INFO - ================================================================================
2025-07-15 05:21:58,112 - INFO - 📁 Processing folder: /home/ubuntu/pdf/
2025-07-15 05:21:58,112 - INFO - 🎯 Target index: genomics-publications
2025-07-15 05:21:58,112 - INFO - 🔬 Metadata extraction: true
2025-07-15 05:21:58,112 - INFO - 🌐 Crossref enrichment: true
2025-07-15 05:21:58,112 - INFO - 🔧 Initializing enhanced pipeline...
2025-07-15 05:21:59,096 - INFO - HTTP Request: GET https://api.openai.com/v1/models "HTTP/1.1 200 OK"
2025-07-15 05:21:59,115 - INFO - ✅ OpenAI client (new) initialized successfully
2025-07-15 05:21:59,456 - INFO - ✅ Pinecone client (new) initialized successfully
2025-07-15 05:21:59,457 - INFO - 🔍 Found 3 PDF files in /home/ubuntu/pdf/
2025-07-15 05:21:59,457 - INFO - 
📁 File 1/3
2025-07-15 05:21:59,459 - INFO - 
================================================================================
2025-07-15 05:21:59,459 - INFO - 🔄 Processing: t2d.pdf
2025-07-15 05:21:59,459 - INFO - ================================================================================
2025-07-15 05:21:59,460 - INFO - 📄 Extracting text from: t2d.pdf
2025-07-15 05:22:03,930 - INFO - ✅ Extracted text using pdfplumber: 156746 chars
2025-07-15 05:22:03,930 - INFO - 🔍 Extracting metadata from: t2d.pdf
2025-07-15 05:22:05,100 - INFO - 📄 Found DOI: 10.1016/j.jmb.2019.12.045
2025-07-15 05:22:05,101 - INFO - 📰 Found journal: 1552 IsletBiology in Type2Diabetes
2025-07-15 05:22:05,101 - INFO - 📅 Found year: 2020
2025-07-15 05:22:05,103 - INFO - 📝 Found abstract: 1273 chars
2025-07-15 05:22:05,103 - INFO - 📋 Extracted metadata: DOI=10.1016/j.jmb.2019.12.045, Authors=1, Journal=1552 IsletBiology in Type2Diabetes, Year=2020
2025-07-15 05:22:05,103 - INFO - 🌐 Enriching with Crossref: 10.1016/j.jmb.2019.12.045
2025-07-15 05:22:05,518 - INFO - ✅ Crossref enriched: journal=Journal of Molecular Biology, authors=2, citations=30
2025-07-15 05:22:05,519 - INFO - 🌐 Enhanced with Crossref: citations=30
2025-07-15 05:22:05,521 - INFO - 🧩 Created 21 chunks for document d93e839185070a6b
2025-07-15 05:22:05,521 - INFO - 🤖 Generating embeddings for 21 texts...
2025-07-15 05:22:06,724 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-07-15 05:22:06,829 - INFO - ✅ Generated 21 embeddings
2025-07-15 05:22:06,833 - INFO - 📤 Uploading 21 vectors to Pinecone...
2025-07-15 05:22:07,840 - INFO - ✅ Uploaded batch 1/1 (21 vectors)
2025-07-15 05:22:07,840 - INFO - 🎉 Successfully uploaded all 21 vectors with enhanced metadata
2025-07-15 05:22:07,840 - INFO - ✅ Successfully processed t2d.pdf
2025-07-15 05:22:07,840 - INFO -    📄 Pages: 28
2025-07-15 05:22:07,840 - INFO -    📝 Words: 18115
2025-07-15 05:22:07,840 - INFO -    🧩 Chunks: 21
2025-07-15 05:22:07,840 - INFO -    👥 Authors: 2
2025-07-15 05:22:07,840 - INFO -    📰 Journal: Journal of Molecular Biology
2025-07-15 05:22:07,840 - INFO -    📅 Year: 2020
2025-07-15 05:22:07,840 - INFO -    📊 Citations: 30
2025-07-15 05:22:07,840 - INFO -    🔗 DOI: 10.1016/j.jmb.2019.12.045
2025-07-15 05:22:07,840 - INFO -    🆔 Doc ID: d93e839185070a6b
2025-07-15 05:22:08,341 - INFO - 
📁 File 2/3
2025-07-15 05:22:08,343 - INFO - 
================================================================================
2025-07-15 05:22:08,343 - INFO - 🔄 Processing: t1d.pdf
2025-07-15 05:22:08,343 - INFO - ================================================================================
2025-07-15 05:22:08,345 - INFO - 📄 Extracting text from: t1d.pdf
2025-07-15 05:22:13,305 - INFO - ✅ Extracted text using pdfplumber: 133159 chars
2025-07-15 05:22:13,306 - INFO - 🔍 Extracting metadata from: t1d.pdf
2025-07-15 05:22:14,493 - INFO - 📄 Found DOI: 10.1016/j.molmet.2024.101973
2025-07-15 05:22:14,496 - INFO - 📝 Found abstract: 1282 chars
2025-07-15 05:22:14,496 - INFO - 📋 Extracted metadata: DOI=10.1016/j.molmet.2024.101973, Authors=1, Journal=None, Year=None
2025-07-15 05:22:14,496 - INFO - 🌐 Enriching with Crossref: 10.1016/j.molmet.2024.101973
2025-07-15 05:22:14,834 - INFO - ✅ Crossref enriched: journal=Molecular Metabolism, authors=12, citations=4
2025-07-15 05:22:14,834 - INFO - 🌐 Enhanced with Crossref: citations=4
2025-07-15 05:22:14,836 - INFO - 🧩 Created 18 chunks for document 39340c53332b9df1
2025-07-15 05:22:14,836 - INFO - 🤖 Generating embeddings for 18 texts...
2025-07-15 05:22:15,983 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-07-15 05:22:15,996 - INFO - ✅ Generated 18 embeddings
2025-07-15 05:22:15,997 - INFO - 📤 Uploading 18 vectors to Pinecone...
2025-07-15 05:22:16,569 - INFO - ✅ Uploaded batch 1/1 (18 vectors)
2025-07-15 05:22:16,569 - INFO - 🎉 Successfully uploaded all 18 vectors with enhanced metadata
2025-07-15 05:22:16,570 - INFO - ✅ Successfully processed t1d.pdf
2025-07-15 05:22:16,570 - INFO -    📄 Pages: 21
2025-07-15 05:22:16,570 - INFO -    📝 Words: 10193
2025-07-15 05:22:16,570 - INFO -    🧩 Chunks: 18
2025-07-15 05:22:16,570 - INFO -    👥 Authors: 12
2025-07-15 05:22:16,570 - INFO -    📰 Journal: Molecular Metabolism
2025-07-15 05:22:16,570 - INFO -    📅 Year: 2024
2025-07-15 05:22:16,570 - INFO -    📊 Citations: 4
2025-07-15 05:22:16,570 - INFO -    🔗 DOI: 10.1016/j.molmet.2024.101973
2025-07-15 05:22:16,570 - INFO -    🆔 Doc ID: 39340c53332b9df1
2025-07-15 05:22:17,070 - INFO - 
📁 File 3/3
2025-07-15 05:22:17,183 - INFO - 
================================================================================
2025-07-15 05:22:17,183 - INFO - 🔄 Processing: db230130.pdf
2025-07-15 05:22:17,183 - INFO - ================================================================================
2025-07-15 05:22:17,295 - INFO - 📄 Extracting text from: db230130.pdf
2025-07-15 05:22:28,856 - INFO - ✅ Extracted text using pdfplumber: 49756 chars
2025-07-15 05:22:28,857 - INFO - 🔍 Extracting metadata from: db230130.pdf
2025-07-15 05:22:29,674 - INFO - 📄 Found DOI: 10.2337/db23-0130
2025-07-15 05:22:29,674 - INFO - 📰 Found journal: by the American Diabetes Association. Readers may use this article
2025-07-15 05:22:29,674 - INFO - 📅 Found year: 2023
2025-07-15 05:22:29,677 - INFO - 📋 Extracted metadata: DOI=10.2337/db23-0130, Authors=0, Journal=by the American Diabetes Association. Readers may use this article, Year=2023
2025-07-15 05:22:29,677 - INFO - 🌐 Enriching with Crossref: 10.2337/db23-0130
2025-07-15 05:22:29,958 - INFO - ✅ Crossref enriched: journal=Diabetes, authors=7, citations=37
2025-07-15 05:22:29,959 - INFO - 🌐 Enhanced with Crossref: citations=37
2025-07-15 05:22:29,959 - INFO - 🧩 Created 7 chunks for document 8245737551313d2f
2025-07-15 05:22:29,959 - INFO - 🤖 Generating embeddings for 7 texts...
2025-07-15 05:22:30,619 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-07-15 05:22:30,621 - INFO - ✅ Generated 7 embeddings
2025-07-15 05:22:30,622 - INFO - 📤 Uploading 7 vectors to Pinecone...
2025-07-15 05:22:30,998 - INFO - ✅ Uploaded batch 1/1 (7 vectors)
2025-07-15 05:22:30,999 - INFO - 🎉 Successfully uploaded all 7 vectors with enhanced metadata
2025-07-15 05:22:30,999 - INFO - ✅ Successfully processed db230130.pdf
2025-07-15 05:22:30,999 - INFO -    📄 Pages: 10
2025-07-15 05:22:30,999 - INFO -    📝 Words: 5001
2025-07-15 05:22:30,999 - INFO -    🧩 Chunks: 7
2025-07-15 05:22:30,999 - INFO -    👥 Authors: 7
2025-07-15 05:22:30,999 - INFO -    📰 Journal: Diabetes
2025-07-15 05:22:30,999 - INFO -    📅 Year: 2023
2025-07-15 05:22:30,999 - INFO -    📊 Citations: 37
2025-07-15 05:22:30,999 - INFO -    🔗 DOI: 10.2337/db23-0130
2025-07-15 05:22:30,999 - INFO -    🆔 Doc ID: 8245737551313d2f
2025-07-15 05:22:31,500 - INFO - 
==========================================================================================
2025-07-15 05:22:31,500 - INFO - 🎯 ENHANCED PROCESSING SUMMARY
2025-07-15 05:22:31,500 - INFO - ==========================================================================================
2025-07-15 05:22:31,500 - INFO - 📁 Folder: /home/ubuntu/pdf/
2025-07-15 05:22:31,500 - INFO - 📊 Total files: 3
2025-07-15 05:22:31,500 - INFO - ✅ Successfully processed: 3
2025-07-15 05:22:31,500 - INFO - ❌ Failed: 0
2025-07-15 05:22:31,500 - INFO - ⏭️ Skipped (already processed): 0
2025-07-15 05:22:31,500 - INFO - 🧩 Total chunks created: 46
2025-07-15 05:22:31,500 - INFO - 🔗 DOIs found: 3
2025-07-15 05:22:31,500 - INFO - 🌐 Crossref enriched: 3
2025-07-15 05:22:31,500 - INFO - 📊 Total citations tracked: 71
2025-07-15 05:22:31,500 - INFO - 📈 Success rate: 100.0%
2025-07-15 05:22:31,500 - INFO - ==========================================================================================
2025-07-15 05:22:31,500 - INFO - 
📋 Generating processing summary...
2025-07-15 05:22:31,500 - INFO - 
📊 PROCESSING SUMMARY:
2025-07-15 05:22:31,500 - INFO -    Files processed: 3
2025-07-15 05:22:31,500 - INFO -    Total chunks: 46
2025-07-15 05:22:31,500 - INFO -    Total words: 33,309
2025-07-15 05:22:31,500 - INFO -    Total citations: 71
2025-07-15 05:22:31,500 - INFO -    Metadata coverage:
2025-07-15 05:22:31,500 - INFO -      DOIs: 100.0%
2025-07-15 05:22:31,500 - INFO -      Journals: 100.0%
2025-07-15 05:22:31,500 - INFO -      Authors: 100.0%
2025-07-15 05:22:31,500 - INFO -    Top journals:
2025-07-15 05:22:31,500 - INFO -      Journal of Molecular Biology: 1 papers
2025-07-15 05:22:31,500 - INFO -      Molecular Metabolism: 1 papers
2025-07-15 05:22:31,500 - INFO -      Diabetes: 1 papers
2025-07-15 05:22:31,500 - INFO - 🎉 All files processed successfully!
2025-07-15 05:22:31,500 - INFO - 
💡 Next steps:
2025-07-15 05:22:31,500 - INFO -    - Test enhanced search with filters
2025-07-15 05:22:31,501 - INFO -    - Query by author, journal, or year
2025-07-15 05:22:31,501 - INFO -    - Analyze your research collection
```
```
Core Scripts (Required):

enhanced_pdf_pipeline.py - Main enhanced pipeline (root directory)
enhanced_search.py - Search module (root directory)
scripts/analytics.py - Analytics and reporting
scripts/test_enhanced_search.py - Search testing suite

Setup & Migration Scripts (Optional but Helpful):

scripts/setup_environment.py - Environment setup
scripts/migrate_existing_data.py - Migrate existing vectors

Usage Examples:
# Setup environment
python scripts/setup_environment.py

# Process PDFs with enhanced metadata
python enhanced_pdf_pipeline.py ./pdfs $OPENAI_API_KEY $PINECONE_API_KEY genomics-publications

# Run analytics
python scripts/analytics.py
python scripts/analytics.py --export

# Test search capabilities
python scripts/test_enhanced_search.py
python scripts/test_enhanced_search.py --interactive
python scripts/test_enhanced_search.py --query "CRISPR" --journal "Nature" --year "2020-2024"

# Migrate existing data (if needed)
python scripts/migrate_existing_data.py
python scripts/migrate_existing_data.py --analyze
```
# Enhanced PDF Processing Pipeline with Rich Metadata

## 🎯 **What This Enhanced Pipeline Does:**

1. **Processes PDFs with Rich Metadata** - Extracts DOIs, authors, journals, citations, and more
2. **Crossref API Integration** - Enriches metadata with authoritative publication data
3. **Advanced Text Processing** - Intelligent chunking with type classification (abstract, methods, results)
4. **Semantic Search Ready** - Creates embeddings with enhanced metadata for powerful filtering
5. **Analytics & Reporting** - Comprehensive insights into your research collection
6. **Production Features** - Duplicate detection, resume capability, and detailed logging

## 🚀 **Quick Setup (5 minutes):**

```bash
# 1. Install dependencies
pip install PyPDF2==3.0.1 pdfplumber==0.10.3 pinecone-client==2.2.4 openai==1.3.5 python-dotenv==1.0.0 requests==2.31.0

# 2. Set up environment (optional but recommended)
python scripts/setup_environment.py

# 3. Configure your API keys
export OPENAI_API_KEY="sk-your-key-here"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_INDEX_NAME="genomics-publications"

# 4. Run enhanced processing
python enhanced_pdf_pipeline.py /path/to/your/pdfs $OPENAI_API_KEY $PINECONE_API_KEY genomics-publications
```

## 📊 **Enhanced Features:**

### Rich Metadata Extraction
- **DOI Detection** - Automatically finds and validates DOIs
- **Author Extraction** - Parses author names and affiliations
- **Journal Identification** - Extracts publication venues
- **Year Detection** - Identifies publication years
- **Institution Mapping** - Links papers to research institutions
- **Abstract Extraction** - Captures paper summaries

### Crossref API Integration
- **Citation Counts** - Real-time citation metrics
- **Authoritative Data** - Verified publication information
- **Publisher Details** - Publisher and ISSN information
- **Subject Classification** - Research area categorization
- **Impact Metrics** - Citation-based impact assessment

### Advanced Search Capabilities
- **Multi-field Filtering** - Search by author, journal, year, citations
- **Semantic Search** - Content-based similarity matching
- **Chunk Type Filtering** - Target specific paper sections
- **Institution Search** - Find papers by research organization
- **High-Impact Discovery** - Filter by citation thresholds

## 🔧 **Usage Examples:**

### Basic Processing
```bash
# Process all PDFs in a folder
python pdf_ingestion_pipeline.py ./research_papers $OPENAI_API_KEY $PINECONE_API_KEY my-index

# With custom settings
EXTRACT_PDF_METADATA=true ENRICH_WITH_CROSSREF=true python enhanced_pdf_pipeline.py ./pdfs ...
```

### Advanced Search
```python
from enhanced_search import EnhancedVectorSearch

# Initialize search
searcher = EnhancedVectorSearch()

# Search with filters
results = searcher.search_by_text(
    query_text="CRISPR gene editing",
    journal="Nature",
    year_range=(2020, 2024),
    min_citations=10,
    chunk_type="abstract"
)

# Specialized searches
high_impact = searcher.get_high_impact_papers(min_citations=50)
by_author = searcher.find_similar_papers_by_author("Smith", "machine learning")
abstracts = searcher.search_abstracts_only("single cell sequencing")
```

### Analytics & Reporting
```bash
# View collection analytics
python scripts/analytics.py

# Export detailed report
python scripts/analytics.py --export

# Interactive search testing
python scripts/test_enhanced_search.py --interactive

# Single query with filters
python scripts/test_enhanced_search.py --query "RNA sequencing" --journal "Cell" --year "2023"
```

## 📈 **Analytics Dashboard:**

The enhanced pipeline provides comprehensive analytics:

- **Collection Overview** - Total papers, citations, coverage metrics
- **Journal Analysis** - Publication venue distribution
- **Author Insights** - Most prolific researchers
- **Institution Mapping** - Research organization analysis
- **Temporal Trends** - Publication year distribution
- **Impact Assessment** - Citation statistics and high-impact papers
- **Metadata Quality** - DOI coverage, enrichment success rates

## 🎯 **Advanced Configuration:**

### Environment Variables
```bash
# Metadata extraction settings
export EXTRACT_PDF_METADATA=true
export ENRICH_WITH_CROSSREF=true
export CROSSREF_EMAIL=your-email@domain.com

# Performance tuning
export CROSSREF_DELAY=0.5
export MAX_CONCURRENT_REQUESTS=5
```

### Chunk Configuration
```python
# In enhanced_pdf_pipeline.py
chunk_size = 8000  # Adjust chunk size
overlap = 200      # Overlap between chunks
```

### Search Customization
```python
# Custom filters
results = searcher.search_with_filters(
    query_vector=embedding,
    author="Jane Doe",
    journal="Science",
    year_range=(2020, 2024),
    keywords=["CRISPR", "gene editing"],
    institution="MIT",
    chunk_type="methods",
    min_citations=5,
    publisher="Nature Publishing Group",
    top_k=20
)
```

## 🔄 **Migration from Basic Pipeline:**

If you have existing vectors, migrate them to the enhanced format:

```bash
# Analyze current metadata
python scripts/migrate_existing_data.py --analyze

# Migrate existing data
python scripts/migrate_existing_data.py
```

## 🚨 **Troubleshooting:**

### Common Issues
1. **API Rate Limits** - Increase `CROSSREF_DELAY` or reduce `MAX_CONCURRENT_REQUESTS`
2. **Memory Issues** - Reduce chunk size or process smaller batches
3. **PDF Parsing Errors** - Pipeline automatically falls back to alternative extraction methods
4. **Metadata Gaps** - Enable both PDF extraction and Crossref enrichment

### Performance Optimization
- **Batch Processing** - Pipeline automatically handles large collections
- **Resume Capability** - Safely interrupt and restart processing
- **Duplicate Detection** - Skips already processed files
- **Caching** - Crossref results are cached during session

## 📋 **Output Examples:**

### Processing Summary
```
🎯 ENHANCED PROCESSING SUMMARY
===============================
📁 Folder: /research_papers/
📊 Total files: 150
✅ Successfully processed: 147
❌ Failed: 3
⏭️ Skipped (already processed): 25
🧩 Total chunks created: 3,241
🔗 DOIs found: 142
🌐 Crossref enriched: 138
📊 Total citations tracked: 15,420
📈 Success rate: 98.0%
```

### Analytics Report
```
📊 Vector Store Analytics
========================
Total vectors: 3,241
Journal coverage: 94.2%
Year coverage: 96.8%
Author coverage: 89.5%
DOI coverage: 95.3%

📰 Top Journals (15 total)
--------------------------
Nature                     42 papers (12.1%)
Science                    38 papers (11.0%)
Cell                       31 papers (8.9%)
Nature Medicine           24 papers (6.9%)
```

## 🌟 **Key Benefits:**

- **Rich Metadata** - Comprehensive paper information for advanced search
- **Citation Tracking** - Real-time impact metrics via Crossref
- **Intelligent Chunking** - Section-aware text processing
- **Advanced Filtering** - Multi-dimensional search capabilities
- **Production Ready** - Robust error handling and logging
- **Scalable Architecture** - Handles large research collections
- **Resume Capability** - Fault-tolerant processing
- **Analytics Insights** - Deep understanding of your research collection

## 🔗 **Integration Examples:**

### Web Application
```python
# Flask/FastAPI integration
from enhanced_search import EnhancedVectorSearch

app = Flask(__name__)
searcher = EnhancedVectorSearch()

@app.route('/search')
def search():
    query = request.args.get('q')
    journal = request.args.get('journal')
    year = request.args.get('year')
    
    results = searcher.search_by_text(
        query_text=query,
        journal=journal,
        year_range=(int(year), int(year)) if year else None
    )
    return jsonify(results)
```

### Research Dashboard
```python
# Analytics dashboard
def get_dashboard_data():
    searcher = EnhancedVectorSearch()
    analytics = searcher.get_analytics()
    
    return {
        'total_papers': analytics['total_vectors'],
        'high_impact': searcher.get_high_impact_papers(min_citations=100),
        'recent_papers': searcher.get_papers_by_year(2024),
        'top_journals': get_journal_distribution()
    }
```

This enhanced pipeline transforms your PDF collection into a powerful, searchable knowledge base with rich metadata and advanced analytics capabilities.

## 🆕 **Section-Aware Chunking and Reference Exclusion**

- The ingestion pipeline now performs **section-aware chunking**:
  - It detects common section headers (e.g., Abstract, Introduction, Methods, Results, Discussion, etc.).
  - **Chunking stops at the References section** (or Bibliography); any content after these headers is excluded from ingestion and embeddings.
  - Each chunk is associated with its section (e.g., 'introduction', 'methods', etc.), improving search and analytics.
- **Chunk Metadata** now includes a `section` field, indicating the section of the paper for each chunk.
- This ensures that **reference lists/citations are never included** in the vector store, and enables more precise, section-based search and filtering.

### Example Chunk Metadata
```json
{
  "id": "...",
  "text": "...",
  "chunk_index": 0,
  "doc_id": "...",
  "title": "...",
  "section": "introduction",
  // ... other metadata fields ...
}
```

## 📊 **Enhanced Features:**

- **Section-aware chunking** (excludes references)
- **Section field in chunk metadata** for advanced filtering
- ... (other features as previously listed)
