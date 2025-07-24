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
