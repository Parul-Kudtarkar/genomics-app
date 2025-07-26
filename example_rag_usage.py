#!/usr/bin/env python3
"""
Enhanced RAG Service Usage Examples
Demonstrates all capabilities of the Genomics RAG system
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Environment variables loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

from services.rag_service import GenomicsRAGService, RAGConfig, create_rag_service

class RAGUsageExamples:
    """Comprehensive examples of RAG service usage"""
    
    def __init__(self):
        self.rag = create_rag_service()
        print("🧬 Genomics RAG Service Examples")
        print("=" * 60)
    
    def example_basic_question_answering(self):
        """Example 1: Basic question answering"""
        print("\n🔍 Example 1: Basic Question Answering")
        print("-" * 50)
        
        questions = [
            "What is CRISPR gene editing?",
            "How does gene therapy work?",
            "What are the applications of genomics in medicine?",
            "Explain the role of epigenetics in disease",
            "What is the difference between DNA and RNA?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Question: {question}")
            
            response = self.rag.ask_question(question, top_k=3)
            
            print(f"   Answer: {response.answer[:200]}...")
            print(f"   Sources: {response.num_sources}")
            print(f"   Confidence: {response.confidence_score:.2f}")
            print(f"   Processing time: {response.processing_time:.2f}s")
            
            if response.sources:
                top_source = response.sources[0]
                print(f"   Top source: {top_source['title']} ({top_source.get('year', 'N/A')})")
    
    def example_advanced_filtering(self):
        """Example 2: Advanced filtering capabilities"""
        print("\n🎯 Example 2: Advanced Filtering")
        print("-" * 50)
        
        # Journal filtering
        print("\n1. High-impact Nature papers:")
        response = self.rag.ask_with_paper_focus(
            question="Latest CRISPR developments in medicine",
            journal="Nature",
            min_citations=50,
            top_k=3
        )
        print(f"   Found {response.num_sources} high-impact Nature papers")
        print(f"   Answer: {response.answer[:150]}...")
        
        # Recent research
        print("\n2. Recent research (2020-2024):")
        response = self.rag.ask_with_paper_focus(
            question="Recent advances in cancer genomics",
            year_range=(2020, 2024),
            top_k=3
        )
        print(f"   Found {response.num_sources} recent papers")
        print(f"   Answer: {response.answer[:150]}...")
        
        # Author-specific research
        print("\n3. Research by specific author:")
        response = self.rag.ask_with_paper_focus(
            question="Gene editing techniques",
            author="Doudna",
            top_k=3
        )
        print(f"   Found {response.num_sources} papers by Doudna")
        print(f"   Answer: {response.answer[:150]}...")
    
    def example_specialized_prompts(self):
        """Example 3: Specialized prompt templates"""
        print("\n📝 Example 3: Specialized Prompts")
        print("-" * 50)
        
        # Methods-focused questions
        print("\n1. Methods-focused questions:")
        methods_questions = [
            "What are the standard protocols for DNA sequencing?",
            "How to perform CRISPR gene editing in the laboratory?",
            "What are the steps for RNA extraction and analysis?"
        ]
        
        for question in methods_questions:
            print(f"\n   Question: {question}")
            response = self.rag.ask_about_methods(question, top_k=3)
            print(f"   Answer: {response.answer[:150]}...")
        
        # Results-focused questions
        print("\n2. Results-focused questions:")
        results_questions = [
            "What are the key findings in cancer genomics research?",
            "What results have been achieved with gene therapy trials?",
            "What are the outcomes of CRISPR clinical studies?"
        ]
        
        for question in results_questions:
            print(f"\n   Question: {question}")
            response = self.rag.ask_about_results(question, top_k=3)
            print(f"   Answer: {response.answer[:150]}...")
    
    def example_comparative_analysis(self):
        """Example 4: Comparative analysis"""
        print("\n⚖️ Example 4: Comparative Analysis")
        print("-" * 50)
        
        comparisons = [
            {
                "question": "Compare different gene editing techniques",
                "approaches": ["CRISPR", "TALEN", "Zinc Finger Nucleases"]
            },
            {
                "question": "Compare DNA sequencing technologies",
                "approaches": ["Sanger Sequencing", "Next-Generation Sequencing", "Third-Generation Sequencing"]
            },
            {
                "question": "Compare gene therapy delivery methods",
                "approaches": ["Viral Vectors", "Non-viral Vectors", "CRISPR-Cas9"]
            }
        ]
        
        for i, comparison in enumerate(comparisons, 1):
            print(f"\n{i}. {comparison['question']}")
            response = self.rag.compare_approaches(
                comparison['question'],
                comparison['approaches'],
                top_k=5
            )
            print(f"   Answer: {response.answer[:200]}...")
            print(f"   Sources: {response.num_sources}")
    
    def example_recent_research_summary(self):
        """Example 5: Recent research summaries"""
        print("\n📊 Example 5: Recent Research Summaries")
        print("-" * 50)
        
        topics = [
            "machine learning in genomics",
            "single-cell sequencing",
            "precision medicine",
            "synthetic biology"
        ]
        
        for topic in topics:
            print(f"\nRecent research in: {topic}")
            response = self.rag.summarize_recent_research(
                topic=topic,
                years_back=3,
                max_papers=5
            )
            print(f"   Summary: {response.answer[:200]}...")
            print(f"   Papers analyzed: {response.num_sources}")
    
    def example_document_specific_search(self):
        """Example 6: Document-specific search"""
        print("\n📄 Example 6: Document-Specific Search")
        print("-" * 50)
        
        # First, get some document IDs from a search
        response = self.rag.ask_question("CRISPR applications", top_k=5)
        
        if response.sources:
            # Use the first document for specific search
            doc_id = response.sources[0].get('doc_id')
            if doc_id:
                print(f"\nSearching within document: {response.sources[0]['title']}")
                
                # Ask specific questions about this document
                specific_questions = [
                    "What are the main findings of this study?",
                    "What methods were used?",
                    "What are the limitations mentioned?"
                ]
                
                for question in specific_questions:
                    print(f"\n   Question: {question}")
                    doc_response = self.rag.search_by_document(doc_id, question, top_k=2)
                    print(f"   Answer: {doc_response.answer[:150]}...")
    
    def example_custom_configuration(self):
        """Example 7: Custom configuration"""
        print("\n⚙️ Example 7: Custom Configuration")
        print("-" * 50)
        
        # Create custom configuration for faster, cheaper responses
        fast_config = RAGConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            default_top_k=3,
            enable_caching=True,
            cache_size=500,
            timeout_seconds=15
        )
        
        fast_rag = create_rag_service(config=fast_config)
        
        print("Using fast configuration (gpt-3.5-turbo, 3 sources):")
        response = fast_rag.ask_question("What is epigenetics?", top_k=3)
        
        print(f"   Model used: {response.model_used}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   Answer: {response.answer[:150]}...")
        
        # Create configuration for detailed analysis
        detailed_config = RAGConfig(
            model_name="gpt-4",
            temperature=0.1,
            default_top_k=8,
            enable_caching=True,
            cache_size=1000,
            timeout_seconds=60
        )
        
        detailed_rag = create_rag_service(config=detailed_config)
        
        print("\nUsing detailed configuration (gpt-4, 8 sources):")
        response = detailed_rag.ask_question("Comprehensive overview of gene therapy", top_k=8)
        
        print(f"   Model used: {response.model_used}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   Sources: {response.num_sources}")
        print(f"   Answer: {response.answer[:200]}...")
    
    def example_performance_analysis(self):
        """Example 8: Performance analysis"""
        print("\n📈 Example 8: Performance Analysis")
        print("-" * 50)
        
        # Get service statistics
        stats = self.rag.get_service_statistics()
        
        print("Service Statistics:")
        print(f"   Vector store: {stats.get('vector_store', {}).get('total_vectors', 0)} vectors")
        print(f"   Cache enabled: {stats.get('cache', {}).get('enabled', False)}")
        print(f"   Cache size: {stats.get('cache', {}).get('size', 0)}/{stats.get('cache', {}).get('max_size', 0)}")
        print(f"   Model: {stats.get('config', {}).get('model', 'unknown')}")
        print(f"   Default top-k: {stats.get('config', {}).get('default_top_k', 0)}")
        
        # Test performance with different query types
        print("\nPerformance Test:")
        test_questions = [
            "What is CRISPR?",
            "How does gene therapy work?",
            "What are the applications of genomics?"
        ]
        
        total_time = 0
        total_sources = 0
        
        for question in test_questions:
            response = self.rag.ask_question(question, top_k=3)
            total_time += response.processing_time
            total_sources += response.num_sources
            print(f"   {question[:30]}...: {response.processing_time:.2f}s, {response.num_sources} sources")
        
        print(f"\n   Average time: {total_time/len(test_questions):.2f}s")
        print(f"   Average sources: {total_sources/len(test_questions):.1f}")
    
    def run_all_examples(self):
        """Run all examples"""
        examples = [
            ("Basic Question Answering", self.example_basic_question_answering),
            ("Advanced Filtering", self.example_advanced_filtering),
            ("Specialized Prompts", self.example_specialized_prompts),
            ("Comparative Analysis", self.example_comparative_analysis),
            ("Recent Research Summary", self.example_recent_research_summary),
            ("Document-Specific Search", self.example_document_specific_search),
            ("Custom Configuration", self.example_custom_configuration),
            ("Performance Analysis", self.example_performance_analysis)
        ]
        
        for example_name, example_func in examples:
            try:
                example_func()
                print(f"\n✅ Completed: {example_name}")
            except Exception as e:
                print(f"\n❌ Failed: {example_name} - {e}")
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed!")
        print("=" * 60)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced RAG Service Usage Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all examples
  python example_rag_usage.py
  
  # Run specific example
  python example_rag_usage.py --example basic_qa
  
  # Save output to file
  python example_rag_usage.py --output results.json

Available Examples:
  - basic_qa: Basic question answering
  - advanced_filtering: Advanced filtering capabilities
  - specialized_prompts: Specialized prompt templates
  - comparative_analysis: Comparative analysis
  - recent_research: Recent research summaries
  - document_search: Document-specific search
  - custom_config: Custom configuration
  - performance: Performance analysis
        """
    )
    
    parser.add_argument(
        '--example',
        choices=['basic_qa', 'advanced_filtering', 'specialized_prompts', 
                'comparative_analysis', 'recent_research', 'document_search',
                'custom_config', 'performance'],
        help='Run specific example'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Create examples instance
    examples = RAGUsageExamples()
    
    if args.example:
        # Run specific example
        example_map = {
            'basic_qa': examples.example_basic_question_answering,
            'advanced_filtering': examples.example_advanced_filtering,
            'specialized_prompts': examples.example_specialized_prompts,
            'comparative_analysis': examples.example_comparative_analysis,
            'recent_research': examples.example_recent_research_summary,
            'document_search': examples.example_document_specific_search,
            'custom_config': examples.example_custom_configuration,
            'performance': examples.example_performance_analysis
        }
        
        example_func = example_map.get(args.example)
        if example_func:
            example_func()
        else:
            print(f"❌ Unknown example: {args.example}")
            return 1
    else:
        # Run all examples
        examples.run_all_examples()
    
    return 0

if __name__ == "__main__":
    exit(main()) 