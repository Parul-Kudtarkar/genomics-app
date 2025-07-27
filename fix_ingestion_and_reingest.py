#!/usr/bin/env python3
"""
Fix Ingestion and Re-ingest Documents
Re-ingest documents with proper content storage in the vector database
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_ingestion_pipeline import TextIngestionPipeline
from xml_ingestion_pipeline import XMLIngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionFixer:
    """Fix ingestion issues and re-ingest documents with proper content storage"""
    
    def __init__(self):
        """Initialize the ingestion fixer"""
        self.text_pipeline = TextIngestionPipeline()
        self.xml_pipeline = XMLIngestionPipeline()
        
        logger.info("✅ Ingestion fixer initialized")
    
    def check_current_content(self):
        """Check what content is currently stored"""
        print("\n🔍 CHECKING CURRENT VECTOR STORE CONTENT")
        print("=" * 60)
        
        try:
            # Import the debug function
            from debug_vector_content import examine_vector_content
            examine_vector_content()
            
        except Exception as e:
            logger.error(f"Error checking content: {e}")
    
    def clear_vector_store(self):
        """Clear the entire vector store (use with caution!)"""
        print("\n⚠️  CLEARING VECTOR STORE")
        print("=" * 60)
        print("This will delete ALL vectors from your Pinecone index!")
        print("Make sure you have backups of your source documents.")
        
        confirm = input("\nAre you sure you want to continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Operation cancelled")
            return False
        
        try:
            # Delete all vectors
            self.text_pipeline.pinecone_index.delete(delete_all=True)
            logger.info("✅ Vector store cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
    
    def find_source_documents(self) -> Dict[str, List[str]]:
        """Find all source documents that were previously ingested"""
        print("\n📁 FINDING SOURCE DOCUMENTS")
        print("=" * 60)
        
        documents = {
            'text_files': [],
            'xml_files': []
        }
        
        # Check processed files
        if os.path.exists('processed_text_files.json'):
            with open('processed_text_files.json', 'r') as f:
                processed_texts = json.load(f)
            
            for text_id, info in processed_texts.items():
                file_path = info.get('file_path', '')
                if file_path and os.path.exists(file_path):
                    documents['text_files'].append(file_path)
                    print(f"📄 Found text file: {file_path}")
        
        if os.path.exists('processed_xml_files.json'):
            with open('processed_xml_files.json', 'r') as f:
                processed_xmls = json.load(f)
            
            for xml_id, info in processed_xmls.items():
                file_path = info.get('file_path', '')
                if file_path and os.path.exists(file_path):
                    documents['xml_files'].append(file_path)
                    print(f"📄 Found XML file: {file_path}")
        
        # Look for common directories
        common_dirs = ['pdf', 'text', 'xml', 'data', 'documents']
        for dir_name in common_dirs:
            if os.path.exists(dir_name):
                print(f"\n🔍 Searching in {dir_name}/ directory...")
                
                # Find text files
                for ext in ['*.txt', '*.text']:
                    text_files = list(Path(dir_name).glob(ext))
                    for text_file in text_files:
                        if str(text_file) not in documents['text_files']:
                            documents['text_files'].append(str(text_file))
                            print(f"📄 Found text file: {text_file}")
                
                # Find XML files
                for ext in ['*.xml', '*.nxml']:
                    xml_files = list(Path(dir_name).glob(ext))
                    for xml_file in xml_files:
                        if str(xml_file) not in documents['xml_files']:
                            documents['xml_files'].append(str(xml_file))
                            print(f"📄 Found XML file: {xml_file}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Text files found: {len(documents['text_files'])}")
        print(f"   XML files found: {len(documents['xml_files'])}")
        
        return documents
    
    def reingest_text_files(self, text_files: List[str]):
        """Re-ingest text files with proper content storage"""
        print(f"\n📄 RE-INGESTING TEXT FILES")
        print("=" * 60)
        
        total_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0
        }
        
        for i, text_file in enumerate(text_files, 1):
            print(f"\n📄 Processing {i}/{len(text_files)}: {os.path.basename(text_file)}")
            
            try:
                stats = self.text_pipeline.ingest_text_file(text_file)
                
                if stats.get('text_processed'):
                    total_stats['files_processed'] += 1
                    total_stats['total_chunks'] += stats.get('chunks_created', 0)
                    print(f"✅ Successfully processed ({stats.get('chunks_created', 0)} chunks)")
                else:
                    total_stats['files_failed'] += 1
                    print(f"❌ Failed to process")
                    
            except Exception as e:
                total_stats['files_failed'] += 1
                logger.error(f"Error processing {text_file}: {e}")
        
        print(f"\n📊 TEXT FILES SUMMARY:")
        print(f"   Files processed: {total_stats['files_processed']}")
        print(f"   Files failed: {total_stats['files_failed']}")
        print(f"   Total chunks: {total_stats['total_chunks']}")
        
        return total_stats
    
    def reingest_xml_files(self, xml_files: List[str]):
        """Re-ingest XML files with proper content storage"""
        print(f"\n📄 RE-INGESTING XML FILES")
        print("=" * 60)
        
        total_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0
        }
        
        for i, xml_file in enumerate(xml_files, 1):
            print(f"\n📄 Processing {i}/{len(xml_files)}: {os.path.basename(xml_file)}")
            
            try:
                stats = self.xml_pipeline.ingest_xml_file(xml_file)
                
                if stats.get('papers_processed', 0) > 0:
                    total_stats['files_processed'] += 1
                    total_stats['total_chunks'] += stats.get('total_chunks', 0)
                    print(f"✅ Successfully processed ({stats.get('total_chunks', 0)} chunks)")
                else:
                    total_stats['files_failed'] += 1
                    print(f"❌ Failed to process")
                    
            except Exception as e:
                total_stats['files_failed'] += 1
                logger.error(f"Error processing {xml_file}: {e}")
        
        print(f"\n📊 XML FILES SUMMARY:")
        print(f"   Files processed: {total_stats['files_processed']}")
        print(f"   Files failed: {total_stats['files_failed']}")
        print(f"   Total chunks: {total_stats['total_chunks']}")
        
        return total_stats
    
    def verify_fix(self):
        """Verify that the content is now properly stored"""
        print(f"\n✅ VERIFYING FIX")
        print("=" * 60)
        
        try:
            # Import and run the debug script
            from debug_vector_content import examine_vector_content
            examine_vector_content()
            
            print(f"\n🎉 VERIFICATION COMPLETE!")
            print("If you see content in the debug output, the fix worked!")
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
    
    def run_complete_fix(self, clear_store: bool = False):
        """Run the complete fix process"""
        print("🔧 COMPLETE INGESTION FIX")
        print("=" * 60)
        
        # Step 1: Check current state
        self.check_current_content()
        
        # Step 2: Clear vector store if requested
        if clear_store:
            if not self.clear_vector_store():
                return
        
        # Step 3: Find source documents
        documents = self.find_source_documents()
        
        if not documents['text_files'] and not documents['xml_files']:
            print("❌ No source documents found!")
            print("Please ensure your source documents are available.")
            return
        
        # Step 4: Re-ingest text files
        if documents['text_files']:
            self.reingest_text_files(documents['text_files'])
        
        # Step 5: Re-ingest XML files
        if documents['xml_files']:
            self.reingest_xml_files(documents['xml_files'])
        
        # Step 6: Verify the fix
        self.verify_fix()
        
        print(f"\n🎉 INGESTION FIX COMPLETE!")
        print("Your documents should now have proper content storage.")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Fix Ingestion and Re-ingest Documents")
    parser.add_argument("--mode", choices=[
        "check", "clear", "find", "reingest-text", "reingest-xml", "verify", "complete"
    ], default="complete", help="Operation mode")
    
    parser.add_argument("--clear-store", action="store_true", help="Clear vector store before re-ingesting")
    parser.add_argument("--text-files", nargs="+", help="Specific text files to re-ingest")
    parser.add_argument("--xml-files", nargs="+", help="Specific XML files to re-ingest")
    
    args = parser.parse_args()
    
    try:
        fixer = IngestionFixer()
        
        if args.mode == "check":
            fixer.check_current_content()
        
        elif args.mode == "clear":
            fixer.clear_vector_store()
        
        elif args.mode == "find":
            fixer.find_source_documents()
        
        elif args.mode == "reingest-text":
            if args.text_files:
                fixer.reingest_text_files(args.text_files)
            else:
                documents = fixer.find_source_documents()
                fixer.reingest_text_files(documents['text_files'])
        
        elif args.mode == "reingest-xml":
            if args.xml_files:
                fixer.reingest_xml_files(args.xml_files)
            else:
                documents = fixer.find_source_documents()
                fixer.reingest_xml_files(documents['xml_files'])
        
        elif args.mode == "verify":
            fixer.verify_fix()
        
        elif args.mode == "complete":
            fixer.run_complete_fix(clear_store=args.clear_store)
        
    except Exception as e:
        logger.error(f"❌ Fix failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 