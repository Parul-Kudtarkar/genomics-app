#!/usr/bin/env python3
"""
Enhanced Academic Paper Pipeline - XML FILE INGESTION ONLY
Complete article extraction from XML files (PubMed, arXiv, bioRxiv, etc.)
"""
import os
import sys
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

# Environment variables loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xml_ingestion_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PaperRecord:
    """Unified paper record from XML files"""
    source: str  # 'pubmed', 'arxiv', 'biorxiv', 'pmc', 'generic'
    id: str
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    journal: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    keywords: List[str]
    publication_date: Optional[str]
    license: Optional[str]
    metadata: Dict[str, Any]

class XMLIngestionPipeline:
    """Enhanced XML file ingestion pipeline for academic papers"""
    
    def __init__(self, openai_key: str = None, pinecone_key: str = None, index_name: str = None):
        # Initialize API keys
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.pinecone_key = pinecone_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY required")
        if not self.pinecone_key:
            raise ValueError("PINECONE_API_KEY required")
        
        # Initialize components
        self.openai_client = self._init_openai()
        self.pinecone_index = self._init_pinecone()
        
        # Processing state
        self.processed_file = 'processed_xml_files.json'
        self.processed_papers = self._load_processed_papers()
        
        logger.info(f"🚀 XML Ingestion Pipeline initialized")
        logger.info(f"   📊 Target index: {self.index_name}")
        logger.info(f"   📄 Supports: PubMed, arXiv, bioRxiv, PMC, and generic XML")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            client = openai.OpenAI(api_key=self.openai_key)
            models = client.models.list()
            logger.info("✅ OpenAI client initialized successfully")
            return client
        except Exception as e:
            raise Exception(f"Could not initialize OpenAI client: {e}")
    
    def _init_pinecone(self):
        """Initialize Pinecone client"""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_key)
            index = pc.Index(self.index_name)
            
            stats = index.describe_index_stats()
            logger.info(f"✅ Pinecone initialized - {stats.total_vector_count:,} vectors")
            return index
        except Exception as e:
            raise Exception(f"Could not initialize Pinecone: {e}")
    
    def _load_processed_papers(self) -> Dict[str, Any]:
        """Load processed papers tracking"""
        try:
            if os.path.exists(self.processed_file):
                with open(self.processed_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load processed papers: {e}")
        return {}
    
    def _save_processed_papers(self):
        """Save processed papers tracking"""
        try:
            with open(self.processed_file, 'w') as f:
                json.dump(self.processed_papers, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed papers: {e}")
    
    def detect_xml_type(self, xml_content: str) -> str:
        """Detect the type of XML file based on content"""
        try:
            # Try to parse the XML
            root = ET.fromstring(xml_content)
            
            # Check for PubMed XML
            if root.tag == 'PubmedArticleSet' or root.find('.//PubmedArticle') is not None:
                return 'pubmed'
            
            # Check for arXiv XML
            if root.tag == 'feed' or root.find('.//entry') is not None:
                return 'arxiv'
            
            # Check for PMC XML (JATS format) - this should be checked before bioRxiv
            if (root.tag == 'article' and 
                (root.find('.//article-meta') is not None or 
                 root.find('.//front') is not None or
                 'jats' in xml_content.lower() or
                 'dtd-version' in root.attrib)):
                return 'pmc'
            
            # Check for bioRxiv XML
            if root.find('.//biorxiv') is not None or 'biorxiv' in xml_content.lower():
                return 'biorxiv'
            
            # Check for generic academic XML
            if root.find('.//abstract') is not None or root.find('.//title') is not None:
                return 'generic'
            
            return 'unknown'
            
        except ET.ParseError:
            logger.warning("Could not parse XML content")
            return 'unknown'
    
    def parse_pubmed_xml(self, xml_content: str) -> List[PaperRecord]:
        """Parse PubMed XML format"""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            # Handle both single article and article set
            articles = root.findall('.//PubmedArticle') or [root.find('.//PubmedArticle')]
            
            for article in articles:
                if article is None:
                    continue
                
                try:
                    # Extract basic metadata
                    medline_citation = article.find('.//MedlineCitation')
                    if medline_citation is None:
                        continue
                    
                    # Extract PMID
                    pmid_elem = medline_citation.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else "unknown"
                    
                    # Extract title
                    title_elem = medline_citation.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "Unknown Title"
                    
                    # Extract authors
                    authors = []
                    author_list = medline_citation.find('.//AuthorList')
                    if author_list is not None:
                        for author in author_list.findall('.//Author'):
                            last_name = author.find('.//LastName')
                            fore_name = author.find('.//ForeName')
                            if last_name is not None:
                                author_name = last_name.text
                                if fore_name is not None:
                                    author_name = f"{fore_name.text} {author_name}"
                                authors.append(author_name)
                    
                    # Extract abstract
                    abstract_elem = medline_citation.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Extract journal info
                    journal_elem = medline_citation.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Extract publication date
                    pub_date_elem = medline_citation.find('.//PubDate/Year')
                    year = int(pub_date_elem.text) if pub_date_elem is not None else None
                    
                    # Extract keywords
                    keywords = []
                    keyword_list = medline_citation.findall('.//Keyword')
                    for keyword in keyword_list:
                        if keyword.text:
                            keywords.append(keyword.text)
                    
                    # Extract DOI from PubmedData
                    doi = None
                    article_ids = article.findall('.//ArticleId')
                    for article_id in article_ids:
                        if article_id.get('IdType') == 'doi':
                            doi = article_id.text
                            break
                    
                    # Combine all text content for full_text
                    full_text_parts = []
                    if title:
                        full_text_parts.append(f"Title: {title}")
                    if abstract:
                        full_text_parts.append(f"Abstract: {abstract}")
                    
                    # Extract additional sections if available
                    for section in medline_citation.findall('.//AbstractText'):
                        if section.text and section.text != abstract:
                            full_text_parts.append(section.text)
                    
                    full_text = "\n\n".join(full_text_parts)
                    
                    paper = PaperRecord(
                        source='pubmed',
                        id=pmid,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        full_text=full_text,
                        journal=journal,
                        year=year,
                        doi=doi,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        keywords=keywords,
                        publication_date=str(year) if year else None,
                        license='PubMed',
                        metadata={
                            'pmid': pmid,
                            'xml_source': 'pubmed'
                        }
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse individual PubMed article: {e}")
                    continue
            
            logger.info(f"✅ Parsed {len(papers)} papers from PubMed XML")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
            return []
    
    def parse_arxiv_xml(self, xml_content: str) -> List[PaperRecord]:
        """Parse arXiv XML format"""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            # Handle both single entry and feed
            entries = root.findall('.//entry') or [root.find('.//entry')]
            
            for entry in entries:
                if entry is None:
                    continue
                
                try:
                    # Extract arXiv ID
                    id_elem = entry.find('.//id')
                    arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else "unknown"
                    
                    # Extract title
                    title_elem = entry.find('.//title')
                    title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else "Unknown Title"
                    
                    # Extract authors
                    authors = []
                    author_elems = entry.findall('.//author/name')
                    for author in author_elems:
                        if author.text:
                            authors.append(author.text)
                    
                    # Extract abstract
                    summary_elem = entry.find('.//summary')
                    abstract = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None else ""
                    
                    # Extract publication date
                    published_elem = entry.find('.//published')
                    published_date = published_elem.text if published_elem is not None else None
                    year = None
                    if published_date:
                        try:
                            year = int(published_date.split('-')[0])
                        except:
                            pass
                    
                    # Extract categories/keywords
                    keywords = []
                    category_elems = entry.findall('.//category')
                    for category in category_elems:
                        if category.get('term'):
                            keywords.append(category.get('term'))
                    
                    # Extract DOI if available
                    doi = None
                    doi_elem = entry.find('.//doi')
                    if doi_elem is not None:
                        doi = doi_elem.text
                    
                    # Combine content for full_text
                    full_text_parts = []
                    if title:
                        full_text_parts.append(f"Title: {title}")
                    if abstract:
                        full_text_parts.append(f"Abstract: {abstract}")
                    
                    # Extract additional content if available
                    for content_elem in entry.findall('.//content'):
                        if content_elem.text and content_elem.text != abstract:
                            full_text_parts.append(content_elem.text)
                    
                    full_text = "\n\n".join(full_text_parts)
                    
                    paper = PaperRecord(
                        source='arxiv',
                        id=arxiv_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        full_text=full_text,
                        journal='arXiv preprint',
                        year=year,
                        doi=doi,
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        keywords=keywords,
                        publication_date=published_date,
                        license='arXiv license',
                        metadata={
                            'arxiv_id': arxiv_id,
                            'categories': keywords,
                            'xml_source': 'arxiv'
                        }
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse individual arXiv entry: {e}")
                    continue
            
            logger.info(f"✅ Parsed {len(papers)} papers from arXiv XML")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return []
    
    def parse_pmc_xml(self, xml_content: str) -> List[PaperRecord]:
        """Parse PMC XML format (JATS)"""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            # Since the root is already an article, use it directly
            if root.tag == 'article':
                articles = [root]
            else:
                # Handle both single article and article set
                articles = root.findall('.//article') or [root.find('.//article')]
                if not articles:
                    # If no article found, treat the root as the article
                    articles = [root] if root.tag == 'article' else []
            
            for article in articles:
                if article is None:
                    continue
                
                try:
                    # Extract article ID from multiple possible locations
                    article_id = article.get('id', 'unknown')
                    if article_id == 'unknown':
                        # Try to find PMC ID
                        pmc_id_elem = article.find('.//article-id[@pub-id-type="pmcid"]')
                        if pmc_id_elem is not None:
                            article_id = pmc_id_elem.text
                        else:
                            # Generate ID from title if available
                            title_elem = article.find('.//article-title')
                            if title_elem is not None:
                                title_text = ''.join(title_elem.itertext()).strip()
                                article_id = hashlib.md5(title_text.encode()).hexdigest()[:16]
                    
                    # Extract title
                    title_elem = article.find('.//article-title')
                    title = ""
                    if title_elem is not None:
                        title = ''.join(title_elem.itertext()).strip()
                    
                    # Extract authors
                    authors = []
                    author_elems = article.findall('.//contrib[@contrib-type="author"]/name')
                    for author in author_elems:
                        given_name = author.find('.//given-names')
                        surname = author.find('.//surname')
                        if given_name is not None and surname is not None:
                            authors.append(f"{given_name.text} {surname.text}")
                        elif surname is not None:
                            authors.append(surname.text)
                        elif given_name is not None:
                            authors.append(given_name.text)
                    
                    # Extract abstract
                    abstract_elem = article.find('.//abstract')
                    abstract = ""
                    if abstract_elem is not None:
                        abstract = ''.join(abstract_elem.itertext()).strip()
                    
                    # Extract journal info
                    journal_elem = article.find('.//journal-title')
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Extract publication date
                    pub_date_elem = article.find('.//pub-date/year')
                    year = None
                    if pub_date_elem is not None:
                        try:
                            year = int(pub_date_elem.text)
                        except:
                            pass
                    
                    # Extract DOI
                    doi_elem = article.find('.//article-id[@pub-id-type="doi"]')
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    # Extract keywords
                    keywords = []
                    keyword_elems = article.findall('.//kwd')
                    for keyword in keyword_elems:
                        if keyword.text:
                            keywords.append(keyword.text)
                    
                    # Extract full text content
                    full_text_parts = []
                    if title:
                        full_text_parts.append(f"Title: {title}")
                    if abstract:
                        full_text_parts.append(f"Abstract: {abstract}")
                    
                    # Extract body content
                    body_elem = article.find('.//body')
                    if body_elem is not None:
                        # Extract all sections except references
                        for section in body_elem.findall('.//sec'):
                            sec_title = section.find('.//title')
                            if sec_title is not None:
                                section_title = ''.join(sec_title.itertext()).strip()
                                # Skip references section
                                if 'reference' not in section_title.lower():
                                    section_text = ''.join(section.itertext()).strip()
                                    if section_text:
                                        full_text_parts.append(f"{section_title}: {section_text}")
                    
                    # Also extract any paragraphs directly in body
                    if body_elem is not None:
                        for p_elem in body_elem.findall('.//p'):
                            p_text = ''.join(p_elem.itertext()).strip()
                            if p_text and len(p_text) > 50:
                                full_text_parts.append(p_text)
                    
                    full_text = "\n\n".join(full_text_parts)
                    
                    paper = PaperRecord(
                        source='pmc',
                        id=article_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        full_text=full_text,
                        journal=journal,
                        year=year,
                        doi=doi,
                        url=f"https://europepmc.org/article/PMC/{article_id}",
                        keywords=keywords,
                        publication_date=str(year) if year else None,
                        license='PMC',
                        metadata={
                            'pmc_id': article_id,
                            'xml_source': 'pmc',
                            'jats_format': True
                        }
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse individual PMC article: {e}")
                    continue
            
            logger.info(f"✅ Parsed {len(papers)} papers from PMC XML")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to parse PMC XML: {e}")
            return []
    
    def parse_generic_xml(self, xml_content: str) -> List[PaperRecord]:
        """Parse generic academic XML format"""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            # Try to find paper-like structures
            paper_elements = root.findall('.//paper') or root.findall('.//article') or root.findall('.//publication') or [root]
            
            for paper_elem in paper_elements:
                if paper_elem is None:
                    continue
                
                try:
                    # Extract title
                    title_elem = paper_elem.find('.//title') or paper_elem.find('.//name')
                    title = ""
                    if title_elem is not None:
                        title = ''.join(title_elem.itertext()).strip()
                    
                    # Extract authors
                    authors = []
                    author_elems = paper_elem.findall('.//author') or paper_elem.findall('.//creator')
                    for author in author_elems:
                        author_text = ''.join(author.itertext()).strip()
                        if author_text:
                            authors.append(author_text)
                    
                    # Extract abstract
                    abstract_elem = paper_elem.find('.//abstract') or paper_elem.find('.//summary')
                    abstract = ""
                    if abstract_elem is not None:
                        abstract = ''.join(abstract_elem.itertext()).strip()
                    
                    # Extract journal
                    journal_elem = paper_elem.find('.//journal') or paper_elem.find('.//venue')
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Extract year
                    year_elem = paper_elem.find('.//year') or paper_elem.find('.//date')
                    year = None
                    if year_elem is not None:
                        try:
                            year_text = year_elem.text
                            year = int(year_text.split('-')[0]) if year_text else None
                        except:
                            pass
                    
                    # Extract DOI
                    doi_elem = paper_elem.find('.//doi') or paper_elem.find('.//identifier')
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    # Extract keywords
                    keywords = []
                    keyword_elems = paper_elem.findall('.//keyword') or paper_elem.findall('.//tag')
                    for keyword in keyword_elems:
                        if keyword.text:
                            keywords.append(keyword.text)
                    
                    # Generate ID
                    paper_id = hashlib.md5(f"{title}{authors}{year}".encode()).hexdigest()[:16]
                    
                    # Combine all text content
                    full_text_parts = []
                    if title:
                        full_text_parts.append(f"Title: {title}")
                    if abstract:
                        full_text_parts.append(f"Abstract: {abstract}")
                    
                    # Extract all text content
                    for elem in paper_elem.iter():
                        if elem.text and elem.text.strip() and elem.tag not in ['title', 'abstract', 'author']:
                            text = elem.text.strip()
                            if len(text) > 50:  # Only significant text
                                full_text_parts.append(text)
                    
                    full_text = "\n\n".join(full_text_parts)
                    
                    paper = PaperRecord(
                        source='generic',
                        id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        full_text=full_text,
                        journal=journal,
                        year=year,
                        doi=doi,
                        url=None,
                        keywords=keywords,
                        publication_date=str(year) if year else None,
                        license='Unknown',
                        metadata={
                            'xml_source': 'generic',
                            'root_tag': root.tag
                        }
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse individual generic paper: {e}")
                    continue
            
            logger.info(f"✅ Parsed {len(papers)} papers from generic XML")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to parse generic XML: {e}")
            return []
    
    def ingest_xml_file(self, xml_file_path: str) -> Dict[str, Any]:
        """
        Ingest papers from an XML file
        
        Args:
            xml_file_path: Path to the XML file
        
        Returns:
            Dict with ingestion statistics
        """
        
        stats = {
            'file': xml_file_path,
            'xml_type': 'unknown',
            'papers_found': 0,
            'papers_processed': 0,
            'papers_skipped': 0,
            'papers_failed': 0,
            'total_chunks': 0
        }
        
        logger.info(f"📄 Ingesting XML file: {xml_file_path}")
        
        try:
            # Read XML file
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Clean up XML content to handle DOCTYPE and processing instructions
            import re
            # Remove DOCTYPE declaration
            xml_content = re.sub(r'<!DOCTYPE[^>]*>', '', xml_content)
            # Remove processing instructions
            xml_content = re.sub(r'<\?[^>]*\?>', '', xml_content)
            # Remove comments
            xml_content = re.sub(r'<!--[^>]*-->', '', xml_content)
            
            # Detect XML type
            xml_type = self.detect_xml_type(xml_content)
            stats['xml_type'] = xml_type
            logger.info(f"🔍 Detected XML type: {xml_type}")
            
            # Parse based on type
            papers = []
            if xml_type == 'pubmed':
                papers = self.parse_pubmed_xml(xml_content)
            elif xml_type == 'arxiv':
                papers = self.parse_arxiv_xml(xml_content)
            elif xml_type == 'pmc':
                papers = self.parse_pmc_xml(xml_content)
            elif xml_type == 'generic':
                papers = self.parse_generic_xml(xml_content)
            else:
                logger.warning(f"⚠️ Unknown XML type: {xml_type}")
                return stats
            
            stats['papers_found'] = len(papers)
            logger.info(f"📊 Found {len(papers)} papers in XML file")
            
            # Process papers
            if papers:
                processing_stats = self.process_papers(papers)
                stats.update(processing_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest XML file {xml_file_path}: {e}")
            stats['papers_failed'] = 1
            return stats
    
    def ingest_xml_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all XML files from a directory
        
        Args:
            directory_path: Path to directory containing XML files
        
        Returns:
            Dict with ingestion statistics
        """
        
        stats = {
            'directory': directory_path,
            'files_processed': 0,
            'files_failed': 0,
            'total_papers': 0,
            'total_processed': 0,
            'total_chunks': 0
        }
        
        logger.info(f"📁 Ingesting XML files from directory: {directory_path}")
        
        try:
            directory = Path(directory_path)
            xml_files = list(directory.glob('*.xml')) + list(directory.glob('*.XML'))
            
            logger.info(f"📄 Found {len(xml_files)} XML files")
            
            for xml_file in xml_files:
                try:
                    file_stats = self.ingest_xml_file(str(xml_file))
                    
                    stats['files_processed'] += 1
                    stats['total_papers'] += file_stats.get('papers_found', 0)
                    stats['total_processed'] += file_stats.get('papers_processed', 0)
                    stats['total_chunks'] += file_stats.get('total_chunks', 0)
                    
                    logger.info(f"✅ Processed {xml_file.name}: {file_stats.get('papers_processed', 0)} papers")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {xml_file.name}: {e}")
                    stats['files_failed'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to process directory {directory_path}: {e}")
            return stats
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            logger.info(f"🤖 Generating embeddings for {len(texts)} texts...")
            
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            embeddings = [emb.embedding for emb in response.data]
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def create_chunks(self, paper: PaperRecord, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create chunks from paper content"""
        # Combine title + abstract + full_text
        content = f"Title: {paper.title}\n\n"
        
        if paper.abstract:
            content += f"Abstract: {paper.abstract}\n\n"
        
        if paper.full_text and paper.full_text != paper.abstract:
            content += f"Content: {paper.full_text}"
        
        chunks = []
        pos = 0
        chunk_index = 0
        paper_id = hashlib.sha256(f"{paper.source}:{paper.id}:{paper.title}".encode()).hexdigest()[:16]
        
        while pos < len(content):
            chunk_end = pos + chunk_size
            
            # Try to break at sentence boundary
            if chunk_end < len(content):
                for i in range(chunk_end, max(pos + chunk_size - 200, pos), -1):
                    if content[i] in '.!?':
                        chunk_end = i + 1
                        break
            
            chunk_text = content[pos:chunk_end].strip()
            
            if chunk_text and len(chunk_text.split()) >= 10:
                chunks.append({
                    'id': f"{paper_id}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'paper_id': paper_id,
                    'source': paper.source
                })
                chunk_index += 1
            
            pos = chunk_end - overlap
            if pos >= len(content):
                break
        
        return chunks
    
    def upload_to_vector_store(self, paper: PaperRecord, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """Upload paper vectors to Pinecone"""
        try:
            vectors = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Create metadata
                metadata = {
                    'source': paper.source,
                    'paper_id': chunk['paper_id'],
                    'title': paper.title[:500],
                    'authors': '; '.join(paper.authors) if paper.authors else '',
                    'journal': paper.journal or '',
                    'year': paper.year,
                    'doi': paper.doi or '',
                    'url': paper.url or '',
                    'keywords': ', '.join(paper.keywords) if paper.keywords else '',
                    'chunk_index': chunk['chunk_index'],
                    'word_count': len(chunk['text'].split()),
                    'license': paper.license or '',
                    'publication_date': paper.publication_date or '',
                    'xml_source': paper.metadata.get('xml_source', 'unknown')
                }
                
                # Remove empty values
                clean_metadata = {k: v for k, v in metadata.items() 
                                if v not in [None, '', 0, []]}
                
                vectors.append({
                    'id': chunk['id'],
                    'values': embedding,
                    'metadata': clean_metadata
                })
            
            # Upload to Pinecone
            self.pinecone_index.upsert(vectors=vectors)
            return True
            
        except Exception as e:
            logger.error(f"Vector upload failed: {e}")
            return False
    
    def process_papers(self, papers: List[PaperRecord]) -> Dict[str, int]:
        """Process collected papers"""
        logger.info(f"\n🔄 Processing {len(papers)} papers...")
        
        stats = {
            'total': len(papers),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_chunks': 0
        }
        
        for i, paper in enumerate(papers, 1):
            try:
                logger.info(f"\n📄 Processing paper {i}/{len(papers)}: {paper.title[:50]}...")
                
                # Check if already processed
                paper_key = f"{paper.source}:{paper.id}"
                if paper_key in self.processed_papers:
                    stats['skipped'] += 1
                    logger.info("⏭️ Already processed")
                    continue
                
                # Create chunks
                chunks = self.create_chunks(paper)
                if not chunks:
                    stats['failed'] += 1
                    logger.warning("❌ No chunks created")
                    continue
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.generate_embeddings(texts)
                
                # Upload to vector store
                if self.upload_to_vector_store(paper, chunks, embeddings):
                    stats['processed'] += 1
                    stats['total_chunks'] += len(chunks)
                    
                    # Save progress
                    self.processed_papers[paper_key] = {
                        'title': paper.title,
                        'source': paper.source,
                        'processed_at': datetime.now().isoformat(),
                        'chunk_count': len(chunks)
                    }
                    
                    logger.info(f"✅ Processed successfully ({len(chunks)} chunks)")
                else:
                    stats['failed'] += 1
                    logger.error("❌ Upload failed")
                
                # Save progress periodically
                if i % 10 == 0:
                    self._save_processed_papers()
                
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"❌ Processing failed: {e}")
        
        # Final save
        self._save_processed_papers()
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("🎯 PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Total papers: {stats['total']}")
        logger.info(f"✅ Successfully processed: {stats['processed']}")
        logger.info(f"⏭️ Skipped (already processed): {stats['skipped']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"🧩 Total chunks created: {stats['total_chunks']}")
        logger.info(f"{'='*60}")
        
        return stats

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Academic Paper Pipeline - XML FILE INGESTION ONLY',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single XML file
  python enhanced_pipeline.py ingest-file "path/to/paper.xml"
  
  # Ingest all XML files from a directory
  python enhanced_pipeline.py ingest-dir "path/to/xml/directory"
  
  # Ingest multiple XML files
  python enhanced_pipeline.py ingest-files "file1.xml" "file2.xml" "file3.xml"

Supported XML formats:
  - PubMed XML: PubmedArticleSet format
  - arXiv XML: Atom feed format
  - PMC XML: JATS format
  - Generic XML: Any academic paper XML with title/abstract

Features:
  - Automatic XML format detection
  - Complete article text extraction
  - Reference section filtering
  - Multiple fallback parsing strategies
  - Comprehensive metadata extraction

Environment variables:
  OPENAI_API_KEY=sk-...
  PINECONE_API_KEY=pc-...
  PINECONE_INDEX_NAME=genomics-publications
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest single file command
    file_parser = subparsers.add_parser('ingest-file', help='Ingest a single XML file')
    file_parser.add_argument('xml_file', help='Path to XML file to ingest')
    
    # Ingest directory command
    dir_parser = subparsers.add_parser('ingest-dir', help='Ingest all XML files from a directory')
    dir_parser.add_argument('directory', help='Path to directory containing XML files')
    
    # Ingest multiple files command
    files_parser = subparsers.add_parser('ingest-files', help='Ingest multiple XML files')
    files_parser.add_argument('xml_files', nargs='+', help='Paths to XML files to ingest')
    
    # Common arguments
    for subparser in [file_parser, dir_parser, files_parser]:
        subparser.add_argument('--openai-key', help='OpenAI API key')
        subparser.add_argument('--pinecone-key', help='Pinecone API key')
        subparser.add_argument('--index-name', help='Pinecone index name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    logger.info("🚀 Starting Enhanced Academic Paper Pipeline - XML INGESTION ONLY")
    logger.info("=" * 80)
    logger.info("🎯 XML File Ingestion Mode")
    logger.info("📄 Extracting complete articles from XML files")
    logger.info("🔍 Automatic format detection and parsing")
    
    try:
        # Initialize pipeline
        pipeline = XMLIngestionPipeline(
            openai_key=args.openai_key,
            pinecone_key=args.pinecone_key,
            index_name=args.index_name
        )
        
        if args.command == 'ingest-file':
            # Ingest single XML file
            stats = pipeline.ingest_xml_file(args.xml_file)
            
            # Print ingestion summary
            logger.info(f"\n📊 INGESTION SUMMARY:")
            logger.info(f"   📄 File: {stats['file']}")
            logger.info(f"   🔍 XML Type: {stats['xml_type']}")
            logger.info(f"   📊 Papers found: {stats['papers_found']}")
            logger.info(f"   ✅ Processed: {stats['papers_processed']}")
            logger.info(f"   ⏭️ Skipped: {stats['papers_skipped']}")
            logger.info(f"   ❌ Failed: {stats['papers_failed']}")
            logger.info(f"   🧩 Total chunks: {stats['total_chunks']}")
            
        elif args.command == 'ingest-dir':
            # Ingest all XML files from directory
            stats = pipeline.ingest_xml_directory(args.directory)
            
            # Print ingestion summary
            logger.info(f"\n📊 DIRECTORY INGESTION SUMMARY:")
            logger.info(f"   📁 Directory: {stats['directory']}")
            logger.info(f"   📄 Files processed: {stats['files_processed']}")
            logger.info(f"   ❌ Files failed: {stats['files_failed']}")
            logger.info(f"   📊 Total papers: {stats['total_papers']}")
            logger.info(f"   ✅ Total processed: {stats['total_processed']}")
            logger.info(f"   🧩 Total chunks: {stats['total_chunks']}")
            
        elif args.command == 'ingest-files':
            # Ingest multiple XML files
            total_stats = {
                'files_processed': 0,
                'files_failed': 0,
                'total_papers': 0,
                'total_processed': 0,
                'total_chunks': 0
            }
            
            for xml_file in args.xml_files:
                try:
                    file_stats = pipeline.ingest_xml_file(xml_file)
                    total_stats['files_processed'] += 1
                    total_stats['total_papers'] += file_stats.get('papers_found', 0)
                    total_stats['total_processed'] += file_stats.get('papers_processed', 0)
                    total_stats['total_chunks'] += file_stats.get('total_chunks', 0)
                except Exception as e:
                    logger.error(f"❌ Failed to process {xml_file}: {e}")
                    total_stats['files_failed'] += 1
            
            # Print ingestion summary
            logger.info(f"\n📊 MULTIPLE FILES INGESTION SUMMARY:")
            logger.info(f"   📄 Files processed: {total_stats['files_processed']}")
            logger.info(f"   ❌ Files failed: {total_stats['files_failed']}")
            logger.info(f"   📊 Total papers: {total_stats['total_papers']}")
            logger.info(f"   ✅ Total processed: {total_stats['total_processed']}")
            logger.info(f"   🧩 Total chunks: {total_stats['total_chunks']}")
        
        # Final results
        logger.info("\n💡 Next steps:")
        logger.info("   - Your vector store now contains complete articles from XML")
        logger.info("   - Full text available for AI-powered analysis")
        logger.info("   - Ready for comprehensive search and RAG applications!")
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
