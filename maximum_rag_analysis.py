#!/usr/bin/env python3
"""
Maximum RAG Analysis
Performs RAG analysis with maximum document retrieval for comprehensive answers
"""

import os
import sys
import logging
from typing import Dict, List, Any
from datetime import datetime
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_service import create_rag_service, GenomicsRAGService
from services.search_service import GenomicsSearchService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaximumRAGAnalyzer:
    """Performs comprehensive RAG analysis with maximum document retrieval"""
    
    def __init__(self):
        """Initialize the analyzer"""
        try:
            self.rag_service = create_rag_service()
            self.search_service = GenomicsSearchService(
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            logger.info("✅ Maximum RAG analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer: {e}")
            raise
    
    def analyze_with_maximum_documents(self, question: str, max_documents: int = 50):
        """Analyze a question with maximum document retrieval"""
        print(f"\n🔬 MAXIMUM DOCUMENT ANALYSIS")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Maximum documents: {max_documents}")
        print("-" * 60)
        
        try:
            # Get RAG response with maximum documents
            response = self.rag_service.ask_question(
                question=question,
                top_k=max_documents
            )
            
            print(f"✅ Analysis Complete!")
            print(f"📊 Documents Retrieved: {response.num_sources}")
            print(f"⏱️  Processing Time: {response.processing_time:.2f}s")
            print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
            
            print(f"\n📝 COMPREHENSIVE ANSWER:")
            print("=" * 60)
            print(response.answer)
            
            print(f"\n📚 SOURCE ANALYSIS:")
            print("=" * 60)
            
            # Analyze sources
            journals = {}
            years = {}
            authors = {}
            citation_ranges = {"0-10": 0, "11-50": 0, "51-100": 0, "100+": 0}
            
            for source in response.sources:
                # Journal analysis
                journal = source.get('journal', 'Unknown')
                journals[journal] = journals.get(journal, 0) + 1
                
                # Year analysis
                year = source.get('year', 'Unknown')
                years[year] = years.get(year, 0) + 1
                
                # Author analysis
                for author in source.get('authors', []):
                    authors[author] = authors.get(author, 0) + 1
                
                # Citation analysis
                citations = source.get('citation_count', 0)
                if citations <= 10:
                    citation_ranges["0-10"] += 1
                elif citations <= 50:
                    citation_ranges["11-50"] += 1
                elif citations <= 100:
                    citation_ranges["51-100"] += 1
                else:
                    citation_ranges["100+"] += 1
            
            print(f"📰 Journals ({len(journals)}):")
            for journal, count in sorted(journals.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {journal}: {count} documents")
            
            print(f"\n📅 Years ({len(years)}):")
            for year in sorted(years.keys(), reverse=True)[:10]:
                print(f"  {year}: {years[year]} documents")
            
            print(f"\n👤 Top Authors:")
            for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {author}: {count} documents")
            
            print(f"\n📈 Citation Distribution:")
            for range_name, count in citation_ranges.items():
                print(f"  {range_name} citations: {count} documents")
            
            print(f"\n🔍 TOP SOURCES (by relevance):")
            print("=" * 60)
            for i, source in enumerate(response.sources[:10], 1):
                print(f"{i}. {source.get('title', 'No title')}")
                print(f"   Journal: {source.get('journal', 'Unknown')}")
                print(f"   Year: {source.get('year', 'Unknown')}")
                print(f"   Authors: {', '.join(source.get('authors', [])[:3])}")
                print(f"   Citations: {source.get('citation_count', 0)}")
                print(f"   Relevance: {source.get('relevance_score', 0):.3f}")
                print(f"   Content Type: {source.get('chunk_type', 'Unknown')}")
                print()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in maximum document analysis: {e}")
            return None
    
    def comprehensive_topic_analysis(self, topic: str, max_documents: int = 50):
        """Perform comprehensive analysis of a topic"""
        print(f"\n🎯 COMPREHENSIVE TOPIC ANALYSIS")
        print("=" * 60)
        print(f"Topic: {topic}")
        print(f"Maximum documents: {max_documents}")
        
        # Different types of questions for comprehensive analysis
        analysis_questions = [
            f"What is {topic} and how does it work?",
            f"What are the latest developments in {topic}?",
            f"What are the applications of {topic} in medicine?",
            f"What are the challenges and limitations of {topic}?",
            f"How does {topic} compare to other similar technologies?"
        ]
        
        all_responses = []
        
        for i, question in enumerate(analysis_questions, 1):
            print(f"\n{i}. Analyzing: {question}")
            response = self.analyze_with_maximum_documents(question, max_documents)
            if response:
                all_responses.append(response)
        
        # Summary analysis
        print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total questions analyzed: {len(all_responses)}")
        
        if all_responses:
            total_docs = sum(r.num_sources for r in all_responses)
            avg_confidence = sum(r.confidence_score for r in all_responses) / len(all_responses)
            total_time = sum(r.processing_time for r in all_responses)
            
            print(f"Total documents retrieved: {total_docs}")
            print(f"Average confidence score: {avg_confidence:.3f}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per question: {total_time/len(all_responses):.2f}s")
        
        return all_responses
    
    def compare_with_different_document_counts(self, question: str):
        """Compare RAG performance with different document counts"""
        print(f"\n⚖️ DOCUMENT COUNT COMPARISON")
        print("=" * 60)
        print(f"Question: {question}")
        
        document_counts = [5, 10, 20, 30, 50]
        results = []
        
        for count in document_counts:
            print(f"\n📊 Testing with {count} documents:")
            try:
                response = self.rag_service.ask_question(
                    question=question,
                    top_k=count
                )
                
                results.append({
                    'count': count,
                    'sources': response.num_sources,
                    'confidence': response.confidence_score,
                    'time': response.processing_time,
                    'answer_length': len(response.answer)
                })
                
                print(f"  Documents retrieved: {response.num_sources}")
                print(f"  Confidence score: {response.confidence_score:.3f}")
                print(f"  Processing time: {response.processing_time:.2f}s")
                print(f"  Answer length: {len(response.answer)} characters")
                
            except Exception as e:
                logger.error(f"Error with {count} documents: {e}")
        
        # Summary comparison
        print(f"\n📈 COMPARISON SUMMARY:")
        print("=" * 60)
        print(f"{'Docs':<6} {'Retrieved':<10} {'Confidence':<12} {'Time(s)':<8} {'Length':<8}")
        print("-" * 60)
        for result in results:
            print(f"{result['count']:<6} {result['sources']:<10} {result['confidence']:<12.3f} {result['time']:<8.2f} {result['answer_length']:<8}")
        
        return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Maximum RAG Analysis")
    parser.add_argument("--mode", choices=[
        "single", "topic", "comparison"
    ], default="single", help="Analysis mode")
    
    parser.add_argument("--question", type=str, help="Question to analyze")
    parser.add_argument("--topic", type=str, help="Topic for comprehensive analysis")
    parser.add_argument("--max-docs", type=int, default=50, help="Maximum documents to retrieve")
    
    args = parser.parse_args()
    
    try:
        analyzer = MaximumRAGAnalyzer()
        
        if args.mode == "single":
            if not args.question:
                print("❌ Please provide a question with --question")
                return
            analyzer.analyze_with_maximum_documents(args.question, args.max_docs)
        
        elif args.mode == "topic":
            if not args.topic:
                print("❌ Please provide a topic with --topic")
                return
            analyzer.comprehensive_topic_analysis(args.topic, args.max_docs)
        
        elif args.mode == "comparison":
            if not args.question:
                print("❌ Please provide a question with --question")
                return
            analyzer.compare_with_different_document_counts(args.question)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 