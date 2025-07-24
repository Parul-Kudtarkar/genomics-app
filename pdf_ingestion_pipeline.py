#!/usr/bin/env python3
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
from dataclasses import dataclass
import numpy as np

# PDF processing
import PyPDF2
import pdfplumber

# AI libraries
import openai
from services.section_chunker import SectionAwareChunker
import threading
from functools import wraps
from cachetools import TTLCache

@dataclass
class DuplicateStatus:
    is_duplicate: bool
    confidence: float
    duplicate_of: Optional[str]

class DocumentDeduplicator:
    def __init__(self, similarity_threshold=0.85, metadata_weight=0.3, content_weight=0.7):
        self.similarity_threshold = similarity_threshold
        self.metadata_weight = metadata_weight
        self.content_weight = content_weight

    def detect_duplicates(self, new_doc: Dict, processed_files: Dict) -> DuplicateStatus:
        # DOI exact matching
        doi_match = self._check_doi_duplicates(new_doc, processed_files)
        # Content-based similarity
        content_similarity, duplicate_of_content = self._calculate_content_similarity(new_doc, processed_files)
        # Metadata-based matching
        metadata_similarity, duplicate_of_meta = self._check_metadata_similarity(new_doc, processed_files)
        combined_score = (
            content_similarity * self.content_weight + 
            metadata_similarity * self.metadata_weight
        )
        is_dup = doi_match or combined_score > self.similarity_threshold
        duplicate_of = None
        if doi_match:
            duplicate_of = doi_match
        elif combined_score > self.similarity_threshold:
            duplicate_of = duplicate_of_content or duplicate_of_meta
        return DuplicateStatus(
            is_duplicate=is_dup,
            confidence=max(combined_score, 1.0 if doi_match else 0.0),
            duplicate_of=duplicate_of
        )

    def _check_doi_duplicates(self, new_doc: Dict, processed_files: Dict) -> Optional[str]:
        doi = new_doc.get('doi')
        if not doi:
            return None
        for file_hash, meta in processed_files.items():
            if meta.get('doi') and meta.get('doi') == doi:
                return file_hash
        return None

    def _calculate_content_similarity(self, new_doc: Dict, processed_files: Dict) -> (float, Optional[str]):
        # Use simple Jaccard similarity on text hashes for now
        new_text = new_doc.get('text', '')
        new_set = set(new_text.lower().split())
        best_score = 0.0
        best_file = None
        for file_hash, meta in processed_files.items():
            old_text = meta.get('text', '')
            if not old_text:
                continue
            old_set = set(old_text.lower().split())
            intersection = len(new_set & old_set)
            union = len(new_set | old_set)
            score = intersection / union if union else 0.0
            if score > best_score:
                best_score = score
                best_file = file_hash
        return best_score, best_file

    def _check_metadata_similarity(self, new_doc: Dict, processed_files: Dict) -> (float, Optional[str]):
        # Compare title, authors, year, journal
        new_title = new_doc.get('title', '').lower()
        new_authors = set([a.lower() for a in new_doc.get('authors', [])])
        new_year = new_doc.get('publication_year')
        new_journal = new_doc.get('journal', '').lower()
        best_score = 0.0
        best_file = None
        for file_hash, meta in processed_files.items():
            score = 0.0
            if meta.get('title', '').lower() == new_title and new_title:
                score += 0.4
            if new_year and meta.get('publication_year') == new_year:
                score += 0.2
            meta_authors = set([a.lower() for a in meta.get('authors', [])])
            if new_authors and meta_authors and len(new_authors & meta_authors) > 0:
                score += 0.2
            if meta.get('journal', '').lower() == new_journal and new_journal:
                score += 0.2
            if score > best_score:
                best_score = score
                best_file = file_hash
        return best_score, best_file

class RateLimiter:
    def __init__(self, requests_per_second):
        self.lock = threading.Lock()
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last = time.time()

    def limit(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                elapsed = now - self.last
                self.tokens += elapsed * self.requests_per_second
                if self.tokens > self.requests_per_second:
                    self.tokens = self.requests_per_second
                if self.tokens < 1:
                    sleep_time = (1 - self.tokens) / self.requests_per_second
                    time.sleep(sleep_time)
                    self.tokens = 0
                else:
                    self.tokens -= 1
                self.last = time.time()
            return func(*args, **kwargs)
        return wrapper

class ProductionCrossrefClient:
    def __init__(self, requests_per_second=10, cache_ttl=3600, max_retries=3, backoff_factor=2):
        self.rate_limiter = RateLimiter(requests_per_second)
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def enrich_with_crossref_production(self, doi: str, crossref_email: str) -> Dict[str, Any]:
        @self.rate_limiter.limit
        def _request():
            url = f"https://api.crossref.org/works/{doi}"
            headers = {'User-Agent': f'GenomicsApp/1.0 (mailto:{crossref_email})'}
            return requests.get(url, headers=headers, timeout=15)
        # Check cache first
        if doi in self.cache:
            return self.cache[doi]
        retries = 0
        while retries < self.max_retries:
            try:
                response = _request()
                if response.status_code == 200:
                    data = response.json()
                    work = data.get('message', {})
                    enriched = {
                        'crossref_title': work.get('title', [None])[0],
                        'crossref_journal': work.get('container-title', [None])[0],
                        'crossref_authors': [
                            f"{a.get('given', '')} {a.get('family', '')}".strip()
                            for a in work.get('author', [])[:15]
                            if a.get('family')
                        ],
                        'crossref_year': (work.get('published-print', {}).get('date-parts', [[None]])[0][0]
                                          or work.get('published-online', {}).get('date-parts', [[None]])[0][0]),
                        'citation_count': work.get('is-referenced-by-count', 0),
                        'publisher': work.get('publisher'),
                        'issn': (work.get('ISSN', [None])[0] if work.get('ISSN') else None),
                        'subject_areas': work.get('subject', [])[:10],
                        'crossref_abstract': None
                    }
                    if work.get('abstract'):
                        import re
                        abstract = re.sub(r'<[^>]+>', '', work['abstract'])
                        if len(abstract) > 50:
                            enriched['crossref_abstract'] = abstract
                    self.cache[doi] = enriched
                    return enriched
                elif response.status_code == 404:
                    return {}
                else:
                    time.sleep(self.backoff_factor ** retries)
            except requests.exceptions.Timeout:
                time.sleep(self.backoff_factor ** retries)
            except Exception:
                time.sleep(self.backoff_factor ** retries)
            retries += 1
        return {}

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

@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: float
    issues: List[str]

class ScientificContentValidator:
    def __init__(self):
        self.scientific_terms = self._load_scientific_vocabulary()
        self.protocol_patterns = self._load_protocol_patterns()

    def _load_scientific_vocabulary(self):
        # TODO: Load from file or use a more comprehensive list
        return [
            'gene', 'protein', 'cell', 'assay', 'experiment', 'p-value', 'statistical',
            'PCR', 'sequencing', 'mutation', 'expression', 'analysis', 'protocol', 'method',
            'mouse', 'human', 'zebrafish', 'cancer', 'tumor', 'in vitro', 'in vivo'
        ]

    def _load_protocol_patterns(self):
        # TODO: Use more advanced protocol step patterns
        return ['step', 'incubate', 'centrifuge', 'add', 'mix', 'measure']

    def validate_chunk_quality(self, chunk: Dict) -> ValidationResult:
        issues = []
        text = chunk.get('text', '')
        text_lower = text.lower()
        # Validate scientific content
        if not self._contains_scientific_terminology(text):
            issues.append("Low scientific content density")
        # Validate protocol completeness
        if chunk.get('content_type') == 'protocol_step':
            if not self._is_protocol_actionable(text):
                issues.append("Incomplete protocol information")
        # Validate statistical content
        if chunk.get('has_statistical_data'):
            if not self._validate_statistical_claims(text):
                issues.append("Statistical content lacks proper context")
        quality_score = self._calculate_quality_score(chunk, issues)
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_score=quality_score,
            issues=issues
        )

    def _contains_scientific_terminology(self, text: str) -> bool:
        return any(term in text.lower() for term in self.scientific_terms)

    def _is_protocol_actionable(self, text: str) -> bool:
        return sum(1 for pat in self.protocol_patterns if pat in text.lower()) >= 2

    def _validate_statistical_claims(self, text: str) -> bool:
        # Simple check: must mention a value or method
        import re
        return bool(re.search(r'p[- ]?value\s*[<>=]\s*\d', text.lower()))

    def _calculate_quality_score(self, chunk: Dict, issues: List[str]) -> float:
        base = 1.0
        if issues:
            base -= 0.2 * len(issues)
        if chunk.get('has_statistical_data'):
            base += 0.1
        if chunk.get('methodology_completeness') == 'complete':
            base += 0.1
        return max(0.0, min(1.0, base))

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
        self.section_chunker = SectionAwareChunker()
        self.validator = ScientificContentValidator()
        self.deduplicator = DocumentDeduplicator()
        self.crossref_client = ProductionCrossrefClient()
    
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
                    page_text = page.extract_text()
                    if page_text:
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
        return self.crossref_client.enrich_with_crossref_production(doi, self.crossref_email)
    
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
        """Split text into section-aware chunks for embeddings, excluding References section."""
        import re
        chunk_size = 8000
        overlap = 200

        # Section header patterns (case-insensitive)
        section_patterns = [
            r"^abstract$",
            r"^introduction$",
            r"^background$",
            r"^methods?$",
            r"^materials? and methods$",
            r"^results?$",
            r"^discussion$",
            r"^conclusion$",
            r"^summary$",
            r"^references?$",
            r"^bibliography$",
            r"^acknowledg(e)?ments?$",
        ]
        section_regex = re.compile(r"^\s*([A-Z][A-Za-z0-9 \-]{2,50})\s*$", re.MULTILINE)
        # Clean text
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{2,}', '\n', text)
        text = text.strip()

        # Find all section headers
        matches = list(section_regex.finditer(text))
        section_spans = []
        for i, match in enumerate(matches):
            header = match.group(1).strip().lower()
            # Check if header matches any known section pattern
            for pat in section_patterns:
                if re.match(pat, header, re.IGNORECASE):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    section_spans.append((header, start, end))
                    break

        # If no section headers found, treat whole text as one section
        if not section_spans:
            section_spans = [("full_text", 0, len(text))]

        # Exclude references and anything after
        filtered_sections = []
        for header, start, end in section_spans:
            if re.match(r"references?|bibliography", header, re.IGNORECASE):
                break
            filtered_sections.append((header, start, end))

        chunks = []
        chunk_index = 0
        for section, start, end in filtered_sections:
            section_text = text[start:end].strip()
            if not section_text:
                continue
            # Chunk within section
            pos = 0
            while pos < len(section_text):
                chunk_end = pos + chunk_size
                # Try to break at sentence boundary
                if chunk_end < len(section_text):
                    for i in range(chunk_end, max(pos + chunk_size - 200, pos), -1):
                        if section_text[i] in '.!?':
                            chunk_end = i + 1
                            break
                chunk_text = section_text[pos:chunk_end].strip()
                if chunk_text:
                    chunks.append({
                        'id': f"{doc_id}_chunk_{chunk_index}",
                        'text': chunk_text,
                        'chunk_index': chunk_index,
                        'doc_id': doc_id,
                        'title': title,
                        'section': section
                    })
                    chunk_index += 1
                pos = chunk_end - overlap
                if pos >= len(section_text):
                    break
        self.logger.info(f"🧩 Created {len(chunks)} section-aware chunks for document {doc_id}")
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
    
    def create_enhanced_scientific_metadata(self, chunk: Dict, pdf_metadata: Dict, all_chunks: List[Dict] = None) -> Dict:
        """Create enhanced scientific metadata for a chunk."""
        return {
            # Scientific content classification
            "content_type": self._classify_scientific_content(chunk['text']),
            "section_type": chunk.get('section_type', self._detect_section_type(chunk['text'], chunk.get('chunk_index', 0))),
            "experimental_context": self._extract_experimental_context(chunk['text']),

            # Scientific entity extraction
            "gene_mentions": self._extract_gene_names(chunk['text']),
            "protein_mentions": self._extract_protein_names(chunk['text']),
            "methodology_type": self._classify_methodology(chunk['text']),

            # Content quality indicators
            "has_statistical_data": self._contains_statistical_analysis(chunk['text']),
            "methodology_completeness": self._assess_protocol_completeness(chunk['text']),
            "quantitative_data": self._contains_numerical_data(chunk['text']),

            # Cross-chunk relationships (stub)
            "related_chunks": [],  # TODO: Implement cross-chunk relationship detection
            "prerequisite_chunks": [],  # TODO: Implement dependency detection
            "figure_references": self._extract_figure_references(chunk['text']),

            # Existing metadata (keep all current fields)
            **self._get_existing_metadata(chunk, pdf_metadata)
        }

    def _classify_scientific_content(self, text: str) -> str:
        # TODO: Use ML/NLP for better classification
        text_lower = text.lower()
        if any(word in text_lower for word in ['protocol', 'step', 'procedure']):
            return 'protocol_step'
        if any(word in text_lower for word in ['result', 'data', 'analysis']):
            return 'result'
        if any(word in text_lower for word in ['introduction', 'background']):
            return 'introduction'
        if any(word in text_lower for word in ['discussion', 'conclusion']):
            return 'discussion'
        return 'content'

    def _detect_section_type(self, text: str, chunk_index: int) -> str:
        # Use chunk's section_type if available, else fallback
        text_lower = text.lower()
        if chunk_index == 0 and any(word in text_lower for word in ['abstract', 'summary']):
            return 'abstract'
        for section in ['methods', 'results', 'discussion', 'conclusion', 'references']:
            if section in text_lower:
                return section
        return 'content'

    def _extract_experimental_context(self, text: str) -> str:
        # TODO: Use NER for organism/cell line/disease extraction
        # Simple regex for common terms
        context_terms = ['mouse', 'human', 'cell line', 'zebrafish', 'cancer', 'tumor', 'yeast', 'e. coli']
        found = [term for term in context_terms if term in text.lower()]
        return ', '.join(found) if found else None

    def _extract_gene_names(self, text: str) -> list:
        # TODO: Use gene NER
        import re
        # Simple regex for gene symbols (all caps, 2-8 letters/numbers)
        return re.findall(r'\b[A-Z0-9]{2,8}\b', text)

    def _extract_protein_names(self, text: str) -> list:
        # TODO: Use protein NER
        # Simple stub: look for 'protein' followed by a word
        import re
        return re.findall(r'protein\s+([A-Za-z0-9-]+)', text, re.IGNORECASE)

    def _classify_methodology(self, text: str) -> str:
        # TODO: Use ML for better classification
        methods = ['PCR', 'sequencing', 'Western blot', 'flow cytometry', 'mass spectrometry']
        for method in methods:
            if method.lower() in text.lower():
                return method
        return None

    def _contains_statistical_analysis(self, text: str) -> bool:
        # Simple check for statistical terms
        stats_terms = ['p-value', 'statistical significance', 'confidence interval', 'ANOVA', 't-test']
        return any(term in text.lower() for term in stats_terms)

    def _assess_protocol_completeness(self, text: str) -> str:
        # TODO: Use more advanced protocol completeness checks
        required_terms = ['step', 'incubate', 'centrifuge', 'add', 'mix', 'measure']
        found = [term for term in required_terms if term in text.lower()]
        return 'complete' if len(found) >= 3 else 'incomplete'

    def _contains_numerical_data(self, text: str) -> bool:
        import re
        return bool(re.search(r'\d+\.\d+|\d+%', text))

    def _extract_figure_references(self, text: str) -> list:
        import re
        return re.findall(r'(Figure|Fig\.?|Table)\s*\d+', text)

    def _get_existing_metadata(self, chunk: Dict, pdf_metadata: Dict) -> Dict:
        # Merge chunk and pdf_metadata, prioritizing chunk fields
        meta = dict(pdf_metadata)
        meta.update(chunk)
        return meta

    def upload_to_vector_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], 
                             file_metadata: Dict[str, Any], pdf_metadata: Dict[str, Any]) -> bool:
        """Upload chunks and embeddings to Pinecone with enhanced metadata and validation"""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")
            logger.info(f"\U0001F4E4 Uploading {len(chunks)} vectors to Pinecone...")
            vectors = []
            valid_count = 0
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                enhanced_metadata = self.create_enhanced_scientific_metadata(chunk, pdf_metadata, all_chunks=chunks)
                validation = self.validator.validate_chunk_quality(enhanced_metadata)
                if validation.is_valid:
                    valid_count += 1
                    enhanced_metadata['quality_score'] = validation.quality_score
                    vector_data = {
                        'id': chunk['id'],
                        'values': embedding,
                        'metadata': {k: v for k, v in enhanced_metadata.items() if v is not None and v != '' and v != []}
                    }
                    vectors.append(vector_data)
                else:
                    logger.warning(f"Chunk {chunk['id']} failed validation: {validation.issues}")
            if not vectors:
                logger.error("No valid chunks to upload after validation.")
                return False
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
            logger.info(f"🎉 Successfully uploaded {valid_count} validated vectors with enhanced metadata")
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
            
            # Deduplication check
            dedup_status = self.deduplicator.detect_duplicates(
                {
                    'doi': pdf_metadata.get('doi'),
                    'title': pdf_metadata.get('title'),
                    'authors': pdf_metadata.get('authors', []),
                    'publication_year': pdf_metadata.get('publication_year'),
                    'journal': pdf_metadata.get('journal'),
                    'text': extraction_result['text']
                },
                self.processed_files
            )
            if dedup_status.is_duplicate:
                logger.warning(f"⏭️ Duplicate detected for {filename} (confidence={dedup_status.confidence:.2f}, duplicate_of={dedup_status.duplicate_of})")
                return True
            
            # Create document metadata
            doc_id = hashlib.sha256(f"{filename}{file_hash}".encode()).hexdigest()[:16]
            title = pdf_metadata.get('title') or filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            
            # Create chunks (section-aware)
            chunks = self.section_chunker.create_intelligent_chunks(
                extraction_result['text'], doc_id, title
            )
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
    
    def process_folder_with_reliability(self, folder_path: str) -> Dict[str, int]:
        try:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            pdf_files = list(folder.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {folder_path}")
                return {'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0}
            logger.info(f"🔍 Found {len(pdf_files)} PDF files in {folder_path}")
            processing_state = ProcessingState(folder_path)
            processing_state.stats['total'] = len(pdf_files)
            for i, pdf_file in enumerate(pdf_files, 1):
                try:
                    logger.info(f"\n📁 File {i}/{len(pdf_files)}")
                    # Quick duplicate check
                    file_hash = self._get_file_hash(str(pdf_file))
                    if file_hash in self.processed_files:
                        processing_state.stats['skipped'] += 1
                        logger.info(f"⏭️ Skipping {pdf_file.name} (already processed)")
                        continue
                    # Process with comprehensive error handling
                    try:
                        success = self.process_single_pdf(str(pdf_file))
                        processing_state.update(pdf_file, success)
                    except Exception as e:
                        # Retryable if it's a transient error
                        if isinstance(e, RetryableError):
                            processing_state.add_to_retry_queue(pdf_file, e)
                        else:
                            processing_state.mark_failed(pdf_file, e)
                except Exception as e:
                    logger.error(f"❌ Error processing file {pdf_file}: {e}")
                    processing_state.mark_failed(pdf_file, e)
                if processing_state.should_checkpoint():
                    processing_state.save_checkpoint()
                if self.enable_crossref:
                    time.sleep(0.5)
            # Process retry queue
            for pdf_file, err in processing_state.retry_queue:
                try:
                    logger.info(f"🔁 Retrying {pdf_file} after error: {err}")
                    success = self.process_single_pdf(str(pdf_file))
                    processing_state.update(pdf_file, success)
                except Exception as e:
                    logger.error(f"❌ Retry failed for {pdf_file}: {e}")
                    processing_state.mark_failed(pdf_file, e)
            processing_state.save_checkpoint()
            logger.info(f"\n{'='*90}")
            logger.info("🎯 ENHANCED PROCESSING SUMMARY")
            logger.info(f"{'='*90}")
            logger.info(f"📁 Folder: {folder_path}")
            logger.info(f"📊 Total files: {processing_state.stats['total']}")
            logger.info(f"✅ Successfully processed: {processing_state.stats['processed']}")
            logger.info(f"❌ Failed: {processing_state.stats['failed']}")
            logger.info(f"⏭️ Skipped (already processed): {processing_state.stats['skipped']}")
            logger.info(f"{'='*90}")
            return processing_state.get_final_stats()
        except Exception as e:
            logger.error(f"❌ Folder processing failed: {e}")
            # Save catastrophic failure state
            with open('catastrophic_failure.json', 'w') as f:
                import json
                json.dump({'error': str(e)}, f, indent=2)
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

class ProcessingState:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.stats = {
            'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0,
            'total_chunks': 0, 'total_citations': 0, 'dois_found': 0, 'crossref_enriched': 0
        }
        self.retry_queue = []
        self.failed_files = {}
        self.checkpoint_every = 5
        self.counter = 0

    def update(self, pdf_file, success):
        self.counter += 1
        if success:
            self.stats['processed'] += 1
        else:
            self.stats['failed'] += 1

    def add_to_retry_queue(self, pdf_file, error):
        self.retry_queue.append((pdf_file, str(error)))

    def mark_failed(self, pdf_file, error):
        self.failed_files[str(pdf_file)] = str(error)
        self.stats['failed'] += 1

    def should_checkpoint(self):
        return self.counter % self.checkpoint_every == 0

    def save_checkpoint(self):
        # Save stats and failed files
        with open('processing_checkpoint.json', 'w') as f:
            import json
            json.dump({'stats': self.stats, 'failed_files': self.failed_files, 'retry_queue': self.retry_queue}, f, indent=2)

    def get_final_stats(self):
        return self.stats

class RetryableError(Exception):
    pass
class NonRetryableError(Exception):
    pass

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
        
        # Process all PDFs with reliability
        stats = pipeline.process_folder_with_reliability(folder_path)
        
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
