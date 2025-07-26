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
        api_key ="pcsk_3LmMBC_SMBu4HfVQv8AGLW9LeV4VjsyQ6VVNuJnEDYwbVmS3JV4y6v1urWYWsV8YTkW8AU"

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
        api_key = "pcsk_3LmMBC_SMBu4HfVQv8AGLW9LeV4VjsyQ6VVNuJnEDYwbVmS3JV4y6v1urWYWsV8YTkW8AU"
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
