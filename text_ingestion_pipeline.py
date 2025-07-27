#!/usr/bin/env python3
"""
Simple Text Ingestion Pipeline
Ingests text files and stores them in the vector database
"""
import os
import sys
import hashlib
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai

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
        logging.FileHandler('text_ingestion_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TextRecord:
    """Text record from text files"""
    source: str  # 'text', 'pdf_converted', etc.
    id: str
    title: str
    content: str
    file_path: str
    metadata: Dict[str, Any]

class TextIngestionPipeline:
    """Simple text file ingestion pipeline"""
    
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
        self.processed_file = 'processed_text_files.json'
        self.processed_texts = self._load_processed_texts()
        
        logger.info(f"🚀 Text Ingestion Pipeline initialized")
        logger.info(f"   📊 Target index: {self.index_name}")
    
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
    
    def _load_processed_texts(self) -> Dict[str, Any]:
        """Load processed texts tracking"""
        try:
            if os.path.exists(self.processed_file):
                with open(self.processed_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load processed texts: {e}")
        return {}
    
    def _save_processed_texts(self):
        """Save processed texts tracking"""
        try:
            with open(self.processed_file, 'w') as f:
                json.dump(self.processed_texts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed texts: {e}")
    
    def extract_metadata_from_text(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from text content"""
        try:
            # Use filename as title
            title = Path(file_path).stem.replace('_', ' ').replace('-', ' ')
            
            # Try to extract title from first few lines
            lines = content.split('\n')
            for line in lines[:10]:
                line = line.strip()
                if len(line) > 10 and len(line) < 200 and not line.isdigit():
                    # Skip lines that are likely not titles
                    if not any(word in line.lower() for word in ['page', 'abstract', 'introduction', 'doi:', 'http']):
                        title = line
                        break
            
            # Extract year if present
            year = None
            year_match = re.search(r'\b(19|20)\d{2}\b', content[:2000])
            if year_match:
                year = int(year_match.group(0))
            
            # Extract DOI if present
            doi = None
            doi_match = re.search(r'10\.\d{4,}/[-._;()/:\w]+', content)
            if doi_match:
                doi = doi_match.group(0)
            
            # Generate unique ID
            text_id = hashlib.md5(f"{file_path}:{title}".encode()).hexdigest()[:16]
            
            return {
                'title': title,
                'year': year,
                'doi': doi,
                'text_id': text_id,
                'file_path': file_path,
                'content_length': len(content)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return {
                'title': Path(file_path).stem,
                'year': None,
                'doi': None,
                'text_id': hashlib.md5(file_path.encode()).hexdigest()[:16],
                'file_path': file_path,
                'content_length': len(content)
            }
    
    def ingest_text_file(self, text_file_path: str) -> Dict[str, Any]:
        """Ingest a single text file"""
        stats = {
            'file': text_file_path,
            'text_processed': False,
            'chunks_created': 0,
            'vectors_uploaded': 0
        }
        
        logger.info(f"📄 Ingesting text file: {text_file_path}")
        
        try:
            # Read text file
            with open(text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"⚠️ Empty text file: {text_file_path}")
                return stats
            
            # Extract metadata
            metadata = self.extract_metadata_from_text(content, text_file_path)
            
            # Create text record
            text_record = TextRecord(
                source='text',
                id=metadata['text_id'],
                title=metadata['title'],
                content=content,
                file_path=text_file_path,
                metadata=metadata
            )
            
            # Check if already processed
            text_key = f"text:{text_record.id}"
            if text_key in self.processed_texts:
                logger.info("⏭️ Already processed")
                stats['text_processed'] = True
                return stats
            
            # Process the text
            processing_stats = self.process_text_record(text_record)
            stats.update(processing_stats)
            
            if processing_stats['chunks_created'] > 0:
                stats['text_processed'] = True
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest text file {text_file_path}: {e}")
            return stats
    
    def ingest_text_directory(self, directory_path: str) -> Dict[str, Any]:
        """Ingest all text files from a directory"""
        stats = {
            'directory': directory_path,
            'files_found': 0,
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0
        }
        
        logger.info(f"📁 Ingesting text files from directory: {directory_path}")
        
        try:
            directory = Path(directory_path)
            text_files = list(directory.glob('*.txt')) + list(directory.glob('*.TXT'))
            
            logger.info(f"📄 Found {len(text_files)} text files")
            stats['files_found'] = len(text_files)
            
            for text_file in text_files:
                try:
                    file_stats = self.ingest_text_file(str(text_file))
                    
                    if file_stats['text_processed']:
                        stats['files_processed'] += 1
                        stats['total_chunks'] += file_stats.get('chunks_created', 0)
                        logger.info(f"✅ Processed {text_file.name}: {file_stats.get('chunks_created', 0)} chunks")
                    else:
                        stats['files_failed'] += 1
                        logger.warning(f"⚠️ Failed to process {text_file.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {text_file.name}: {e}")
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
    
    def create_chunks(self, text_record: TextRecord, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create chunks from text content"""
        chunks = []
        pos = 0
        chunk_index = 0
        
        while pos < len(text_record.content):
            chunk_end = pos + chunk_size
            
            # Try to break at sentence boundary
            if chunk_end < len(text_record.content):
                for i in range(chunk_end, max(pos + chunk_size - 200, pos), -1):
                    if text_record.content[i] in '.!?':
                        chunk_end = i + 1
                        break
            
            chunk_text = text_record.content[pos:chunk_end].strip()
            
            if chunk_text and len(chunk_text.split()) >= 10:
                chunks.append({
                    'id': f"{text_record.id}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'text_id': text_record.id,
                    'source': text_record.source
                })
                chunk_index += 1
            
            pos = chunk_end - overlap
            if pos >= len(text_record.content):
                break
        
        return chunks
    
    def upload_to_vector_store(self, text_record: TextRecord, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """Upload text vectors to Pinecone"""
        try:
            vectors = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Create metadata
                metadata = {
                    'source': text_record.source,
                    'text_id': chunk['text_id'],
                    'title': text_record.title[:500],
                    'file_path': text_record.file_path,
                    'chunk_index': chunk['chunk_index'],
                    'word_count': len(chunk['text'].split()),
                    'content_length': text_record.metadata['content_length'],
                    'year': text_record.metadata.get('year'),
                    'doi': text_record.metadata.get('doi'),
                    'text_source': 'text_file',
                    'text': chunk['text'],  # Store the actual content
                    'content': chunk['text']  # Also store as 'content' for compatibility
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
    
    def process_text_record(self, text_record: TextRecord) -> Dict[str, int]:
        """Process a single text record"""
        logger.info(f"\n📄 Processing text: {text_record.title[:50]}...")
        
        stats = {
            'chunks_created': 0,
            'vectors_uploaded': 0
        }
        
        try:
            # Create chunks
            chunks = self.create_chunks(text_record)
            if not chunks:
                logger.warning("❌ No chunks created")
                return stats
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Upload to vector store
            if self.upload_to_vector_store(text_record, chunks, embeddings):
                stats['chunks_created'] = len(chunks)
                stats['vectors_uploaded'] = len(chunks)
                
                # Save progress
                text_key = f"text:{text_record.id}"
                self.processed_texts[text_key] = {
                    'title': text_record.title,
                    'source': text_record.source,
                    'processed_at': datetime.now().isoformat(),
                    'chunk_count': len(chunks)
                }
                
                logger.info(f"✅ Processed successfully ({len(chunks)} chunks)")
            else:
                logger.error("❌ Upload failed")
            
            # Save progress
            self._save_processed_texts()
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
        
        return stats

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Simple Text Ingestion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single text file
  python text_ingestion_pipeline.py ingest-file "path/to/paper.txt"
  
  # Ingest all text files from a directory
  python text_ingestion_pipeline.py ingest-dir "path/to/text/directory"
  
  # Ingest multiple text files
  python text_ingestion_pipeline.py ingest-files "file1.txt" "file2.txt" "file3.txt"

Features:
  - Simple text file ingestion
  - Automatic metadata extraction
  - Chunking and vectorization
  - Progress tracking
  - Duplicate detection

Environment variables:
  OPENAI_API_KEY=sk-...
  PINECONE_API_KEY=pc-...
  PINECONE_INDEX_NAME=genomics-publications
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest single file command
    file_parser = subparsers.add_parser('ingest-file', help='Ingest a single text file')
    file_parser.add_argument('text_file', help='Path to text file to ingest')
    
    # Ingest directory command
    dir_parser = subparsers.add_parser('ingest-dir', help='Ingest all text files from a directory')
    dir_parser.add_argument('directory', help='Path to directory containing text files')
    
    # Ingest multiple files command
    files_parser = subparsers.add_parser('ingest-files', help='Ingest multiple text files')
    files_parser.add_argument('text_files', nargs='+', help='Paths to text files to ingest')
    
    # Common arguments
    for subparser in [file_parser, dir_parser, files_parser]:
        subparser.add_argument('--openai-key', help='OpenAI API key')
        subparser.add_argument('--pinecone-key', help='Pinecone API key')
        subparser.add_argument('--index-name', help='Pinecone index name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    logger.info("🚀 Starting Simple Text Ingestion Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = TextIngestionPipeline(
            openai_key=args.openai_key,
            pinecone_key=args.pinecone_key,
            index_name=args.index_name
        )
        
        if args.command == 'ingest-file':
            # Ingest single text file
            stats = pipeline.ingest_text_file(args.text_file)
            
            # Print ingestion summary
            logger.info(f"\n📊 INGESTION SUMMARY:")
            logger.info(f"   📄 File: {stats['file']}")
            logger.info(f"   ✅ Processed: {stats['text_processed']}")
            logger.info(f"   🧩 Chunks created: {stats['chunks_created']}")
            logger.info(f"   📊 Vectors uploaded: {stats['vectors_uploaded']}")
            
        elif args.command == 'ingest-dir':
            # Ingest all text files from directory
            stats = pipeline.ingest_text_directory(args.directory)
            
            # Print ingestion summary
            logger.info(f"\n📊 DIRECTORY INGESTION SUMMARY:")
            logger.info(f"   📁 Directory: {stats['directory']}")
            logger.info(f"   📄 Files found: {stats['files_found']}")
            logger.info(f"   ✅ Files processed: {stats['files_processed']}")
            logger.info(f"   ❌ Files failed: {stats['files_failed']}")
            logger.info(f"   🧩 Total chunks: {stats['total_chunks']}")
            
        elif args.command == 'ingest-files':
            # Ingest multiple text files
            total_stats = {
                'files_processed': 0,
                'files_failed': 0,
                'total_chunks': 0
            }
            
            for text_file in args.text_files:
                try:
                    file_stats = pipeline.ingest_text_file(text_file)
                    if file_stats['text_processed']:
                        total_stats['files_processed'] += 1
                        total_stats['total_chunks'] += file_stats.get('chunks_created', 0)
                    else:
                        total_stats['files_failed'] += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {text_file}: {e}")
                    total_stats['files_failed'] += 1
            
            # Print ingestion summary
            logger.info(f"\n📊 MULTIPLE FILES INGESTION SUMMARY:")
            logger.info(f"   📄 Files processed: {total_stats['files_processed']}")
            logger.info(f"   ❌ Files failed: {total_stats['files_failed']}")
            logger.info(f"   🧩 Total chunks: {total_stats['total_chunks']}")
        
        # Final results
        logger.info("\n💡 Next steps:")
        logger.info("   - Your vector store now contains text content")
        logger.info("   - Ready for search and RAG applications!")
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
