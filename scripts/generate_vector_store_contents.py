#!/usr/bin/env python3
"""
Vector Store Contents Generator
Generates a comprehensive JSON file with all vector store contents for instant frontend loading
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def load_processed_files() -> Dict[str, Any]:
    """Load and process all ingested files"""
    
    papers = []
    statistics = {
        "total_papers": 0,
        "total_chunks": 0,
        "text_papers": 0,
        "pmc_papers": 0,
        "last_updated": None,
        "generated_at": datetime.now().isoformat()
    }
    
    # Read processed text files
    text_file = Path("processed_text_files.json")
    if text_file.exists():
        print(f"📄 Loading text files from {text_file}")
        with open(text_file, 'r') as f:
            text_data = json.load(f)
            for paper_id, paper_info in text_data.items():
                papers.append({
                    "id": paper_id,
                    "title": paper_info.get("title", "Unknown Title"),
                    "source": "text",
                    "chunk_count": paper_info.get("chunk_count", 0),
                    "processed_at": paper_info.get("processed_at", ""),
                    "type": "Text Document",
                    "file_size": paper_info.get("file_size", 0),
                    "word_count": paper_info.get("word_count", 0)
                })
                statistics["text_papers"] += 1
                statistics["total_chunks"] += paper_info.get("chunk_count", 0)
    
    # Read processed XML files
    xml_file = Path("processed_xml_files.json")
    if xml_file.exists():
        print(f"📄 Loading PMC files from {xml_file}")
        with open(xml_file, 'r') as f:
            xml_data = json.load(f)
            for paper_id, paper_info in xml_data.items():
                papers.append({
                    "id": paper_id,
                    "title": paper_info.get("title", "Unknown Title"),
                    "source": "pmc",
                    "chunk_count": paper_info.get("chunk_count", 0),
                    "processed_at": paper_info.get("processed_at", ""),
                    "type": "PubMed Central Article",
                    "journal": paper_info.get("journal", ""),
                    "authors": paper_info.get("authors", []),
                    "year": paper_info.get("year", ""),
                    "doi": paper_info.get("doi", ""),
                    "pmcid": paper_info.get("pmcid", "")
                })
                statistics["pmc_papers"] += 1
                statistics["total_chunks"] += paper_info.get("chunk_count", 0)
    
    # Sort papers by title
    papers.sort(key=lambda x: x["title"].lower())
    
    # Calculate statistics
    statistics["total_papers"] = len(papers)
    if papers:
        statistics["last_updated"] = max([p["processed_at"] for p in papers if p["processed_at"]])
    
    return {
        "papers": papers,
        "statistics": statistics,
        "metadata": {
            "generator_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "total_files_processed": len(papers),
            "file_sources": {
                "text_files": statistics["text_papers"],
                "pmc_files": statistics["pmc_papers"]
            }
        }
    }

def generate_search_index(papers: List[Dict]) -> Dict[str, Any]:
    """Generate a search index for quick paper lookup"""
    
    search_index = {
        "by_title": {},
        "by_source": {"text": [], "pmc": []},
        "by_year": {},
        "by_journal": {},
        "keywords": {}
    }
    
    for paper in papers:
        # Index by title (for quick lookup)
        search_index["by_title"][paper["title"]] = paper["id"]
        
        # Index by source
        search_index["by_source"][paper["source"]].append(paper["id"])
        
        # Index by year (for PMC papers)
        if paper.get("year"):
            year = paper["year"]
            if year not in search_index["by_year"]:
                search_index["by_year"][year] = []
            search_index["by_year"][year].append(paper["id"])
        
        # Index by journal (for PMC papers)
        if paper.get("journal"):
            journal = paper["journal"]
            if journal not in search_index["by_journal"]:
                search_index["by_journal"][journal] = []
            search_index["by_journal"][journal].append(paper["id"])
    
    return search_index

def main():
    """Main function to generate vector store contents"""
    
    print("🚀 Generating Vector Store Contents...")
    print("=" * 50)
    
    try:
        # Load all data
        data = load_processed_files()
        
        # Generate search index
        search_index = generate_search_index(data["papers"])
        data["search_index"] = search_index
        
        # Create output directory
        output_dir = Path("static")
        output_dir.mkdir(exist_ok=True)
        
        # Save comprehensive data
        output_file = output_dir / "vector_store_contents.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save lightweight version for quick loading
        lightweight_data = {
            "statistics": data["statistics"],
            "papers": data["papers"][:10],  # First 10 papers for preview
            "total_papers": len(data["papers"]),
            "generated_at": data["metadata"]["generated_at"]
        }
        
        lightweight_file = output_dir / "vector_store_preview.json"
        with open(lightweight_file, 'w') as f:
            json.dump(lightweight_data, f, indent=2)
        
        # Print summary
        print(f"✅ Generated vector store contents:")
        print(f"   📁 Full data: {output_file}")
        print(f"   📁 Preview: {lightweight_file}")
        print(f"   📊 Total papers: {data['statistics']['total_papers']}")
        print(f"   📄 Text papers: {data['statistics']['text_papers']}")
        print(f"   🔬 PMC papers: {data['statistics']['pmc_papers']}")
        print(f"   🧩 Total chunks: {data['statistics']['total_chunks']}")
        print(f"   ⏰ Generated at: {data['metadata']['generated_at']}")
        
        # File size info
        full_size = output_file.stat().st_size / 1024  # KB
        preview_size = lightweight_file.stat().st_size / 1024  # KB
        print(f"   📦 Full file size: {full_size:.1f} KB")
        print(f"   📦 Preview file size: {preview_size:.1f} KB")
        
        print("\n🎉 Vector store contents generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating vector store contents: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 