#!/usr/bin/env python3
"""
Professional PDF to Text Converter
Uses GROBID and LangChain's Unstructured for high-quality academic document parsing
"""
import os
import sys
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProfessionalPDFConverter:
    """Professional PDF converter using GROBID and Unstructured"""
    
    def __init__(self, use_grobid: bool = True, grobid_url: str = None):
        self.use_grobid = use_grobid
        self.grobid_url = grobid_url or "http://localhost:8070"
        
        # Try to initialize GROBID
        if use_grobid:
            self.grobid_available = self._check_grobid()
            if not self.grobid_available:
                logger.warning("⚠️ GROBID not available, falling back to Unstructured")
                self.use_grobid = False
        
        # Initialize Unstructured
        self.unstructured_available = self._check_unstructured()
        if not self.unstructured_available:
            logger.error("❌ Neither GROBID nor Unstructured available")
            raise Exception("No PDF parsing tools available")
    
    def _check_grobid(self) -> bool:
        """Check if GROBID is available"""
        try:
            import requests
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            if response.status_code == 200:
                logger.info("✅ GROBID is available")
                return True
        except Exception as e:
            logger.warning(f"GROBID not available: {e}")
        return False
    
    def _check_unstructured(self) -> bool:
        """Check if Unstructured is available"""
        try:
            from unstructured.partition.auto import partition
            logger.info("✅ Unstructured is available")
            return True
        except ImportError:
            logger.warning("Unstructured not installed. Install with: pip install 'unstructured[pdf]'")
            return False
    
    def convert_with_grobid(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Convert PDF using GROBID"""
        try:
            import requests
            
            logger.info(f"🔬 Using GROBID to parse: {pdf_path}")
            
            # Prepare the file for upload
            with open(pdf_path, 'rb') as f:
                files = {'input': (Path(pdf_path).name, f, 'application/pdf')}
                
                # Process with GROBID
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                # Parse the TEI XML response
                import xml.etree.ElementTree as ET
                
                # Clean the XML response
                xml_content = response.text
                # Remove any XML declarations that might cause issues
                xml_content = re.sub(r'<\?xml[^>]*\?>', '', xml_content)
                
                try:
                    root = ET.fromstring(xml_content)
                except ET.ParseError as e:
                    logger.error(f"❌ Failed to parse GROBID XML response: {e}")
                    logger.debug(f"XML content preview: {xml_content[:500]}...")
                    return None
                
                # Extract structured content
                result = {
                    'title': '',
                    'authors': [],
                    'abstract': '',
                    'full_text': '',
                    'sections': [],
                    'references': [],
                    'metadata': {}
                }
                
                # Extract title - try multiple paths
                title_paths = [
                    './/titleStmt/title',
                    './/fileDesc/titleStmt/title',
                    './/head[@type="main"]',
                    './/title'
                ]
                
                for title_path in title_paths:
                    title_elem = root.find(title_path)
                    if title_elem is not None:
                        result['title'] = ''.join(title_elem.itertext()).strip()
                        if result['title']:
                            break
                
                # Extract authors - try multiple paths
                author_paths = [
                    './/titleStmt/author',
                    './/fileDesc/titleStmt/author',
                    './/author'
                ]
                
                for author_path in author_paths:
                    for author in root.findall(author_path):
                        author_name = ''.join(author.itertext()).strip()
                        if author_name:
                            result['authors'].append(author_name)
                    if result['authors']:
                        break
                
                # Extract abstract - try multiple paths
                abstract_paths = [
                    './/profileDesc/abstract',
                    './/abstract',
                    './/div[@type="abstract"]'
                ]
                
                for abstract_path in abstract_paths:
                    abstract_elem = root.find(abstract_path)
                    if abstract_elem is not None:
                        result['abstract'] = ''.join(abstract_elem.itertext()).strip()
                        if result['abstract']:
                            break
                
                # Extract full text (body) - try multiple paths
                body_paths = [
                    './/text/body',
                    './/body',
                    './/div[@type="body"]'
                ]
                
                for body_path in body_paths:
                    body_elem = root.find(body_path)
                    if body_elem is not None:
                        # Extract sections
                        for div in body_elem.findall('.//div'):
                            head = div.find('.//head')
                            section_title = ''.join(head.itertext()).strip() if head is not None else ''
                            
                            # Get section content
                            section_content = []
                            for p in div.findall('.//p'):
                                p_text = ''.join(p.itertext()).strip()
                                if p_text:
                                    section_content.append(p_text)
                            
                            if section_title or section_content:
                                result['sections'].append({
                                    'title': section_title,
                                    'content': '\n'.join(section_content)
                                })
                        
                        # Combine all text from paragraphs
                        all_text = []
                        for p in body_elem.findall('.//p'):
                            p_text = ''.join(p.itertext()).strip()
                            if p_text:
                                all_text.append(p_text)
                        
                        result['full_text'] = '\n\n'.join(all_text)
                        
                        if result['full_text']:
                            break
                
                # If no body found, try to extract any text content
                if not result['full_text']:
                    logger.warning("⚠️ No body content found, trying to extract any text...")
                    all_text = []
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            text = elem.text.strip()
                            if len(text) > 20:  # Only significant text
                                all_text.append(text)
                    result['full_text'] = '\n\n'.join(all_text)
                
                # Extract references
                for bibl in root.findall('.//listBibl/biblStruct'):
                    ref_text = ''.join(bibl.itertext()).strip()
                    if ref_text:
                        result['references'].append(ref_text)
                
                # Extract metadata
                for idno in root.findall('.//publicationStmt/idno'):
                    id_type = idno.get('type')
                    if id_type and idno.text:
                        result['metadata'][id_type] = idno.text.strip()
                
                # Log extraction results
                logger.info(f"✅ GROBID extracted: {len(result['full_text'])} chars, {len(result['sections'])} sections")
                logger.info(f"   Title: {result['title'][:50]}..." if result['title'] else "   Title: Not found")
                logger.info(f"   Authors: {len(result['authors'])} found")
                logger.info(f"   Abstract: {len(result['abstract'])} chars" if result['abstract'] else "   Abstract: Not found")
                
                # Only return if we have meaningful content
                if result['full_text'] and len(result['full_text']) > 100:
                    return result
                else:
                    logger.warning("⚠️ GROBID extracted insufficient content, trying Unstructured...")
                    return None
                
            else:
                logger.error(f"❌ GROBID failed with status {response.status_code}")
                logger.debug(f"Response content: {response.text[:500]}...")
                return None
                
        except Exception as e:
            logger.error(f"❌ GROBID conversion failed: {e}")
            return None
    
    def convert_with_unstructured(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Convert PDF using Unstructured"""
        try:
            from unstructured.partition.auto import partition
            from unstructured.documents.elements import Title, NarrativeText, ListItem
            
            logger.info(f"📄 Using Unstructured to parse: {pdf_path}")
            
            # Parse the PDF
            elements = partition(filename=pdf_path)
            
            result = {
                'title': '',
                'authors': [],
                'abstract': '',
                'full_text': '',
                'sections': [],
                'references': [],
                'metadata': {}
            }
            
            # Process elements
            current_section = None
            section_content = []
            
            for element in elements:
                element_text = str(element).strip()
                
                # Extract title (usually first Title element)
                if isinstance(element, Title) and not result['title']:
                    result['title'] = element_text
                    continue
                
                # Extract abstract (look for common patterns)
                if not result['abstract'] and any(word in element_text.lower() for word in ['abstract', 'summary']):
                    result['abstract'] = element_text
                    continue
                
                # Extract authors (look for patterns like "Author1, Author2")
                if not result['authors'] and ',' in element_text and len(element_text.split(',')) <= 10:
                    # Simple heuristic for author detection
                    if any(word in element_text.lower() for word in ['university', 'department', 'institute', 'center']):
                        continue
                    if len(element_text) < 200:  # Reasonable author length
                        result['authors'] = [author.strip() for author in element_text.split(',')]
                        continue
                
                # Handle sections
                if isinstance(element, Title) and len(element_text) < 100:
                    # Save previous section
                    if current_section and section_content:
                        result['sections'].append({
                            'title': current_section,
                            'content': '\n'.join(section_content)
                        })
                    
                    # Start new section
                    current_section = element_text
                    section_content = []
                else:
                    # Add to current section
                    if element_text:
                        section_content.append(element_text)
            
            # Save last section
            if current_section and section_content:
                result['sections'].append({
                    'title': current_section,
                    'content': '\n'.join(section_content)
                })
            
            # Combine all text
            all_text = []
            for element in elements:
                element_text = str(element).strip()
                if element_text:
                    all_text.append(element_text)
            
            result['full_text'] = '\n\n'.join(all_text)
            
            logger.info(f"✅ Unstructured extracted: {len(result['full_text'])} chars, {len(result['sections'])} sections")
            return result
            
        except Exception as e:
            logger.error(f"❌ Unstructured conversion failed: {e}")
            return None
    
    def convert_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Convert PDF using the best available method"""
        try:
            # Try GROBID first if available
            if self.use_grobid and self.grobid_available:
                result = self.convert_with_grobid(pdf_path)
                if result and result['full_text']:
                    return result
            
            # Fall back to Unstructured
            if self.unstructured_available:
                result = self.convert_with_unstructured(pdf_path)
                if result and result['full_text']:
                    return result
            
            logger.error("❌ No conversion method succeeded")
            return None
            
        except Exception as e:
            logger.error(f"❌ PDF conversion failed: {e}")
            return None
    
    def save_as_text(self, result: Dict[str, Any], output_path: str) -> bool:
        """Save conversion result as formatted text file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write title
                if result['title']:
                    f.write(f"TITLE: {result['title']}\n")
                    f.write("=" * 80 + "\n\n")
                
                # Write authors
                if result['authors']:
                    f.write(f"AUTHORS: {', '.join(result['authors'])}\n\n")
                
                # Write abstract
                if result['abstract']:
                    f.write("ABSTRACT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['abstract'] + "\n\n")
                
                # Write full text
                if result['full_text']:
                    f.write("FULL TEXT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['full_text'] + "\n\n")
                
                # Write sections if available
                if result['sections']:
                    f.write("SECTIONS:\n")
                    f.write("-" * 40 + "\n")
                    for section in result['sections']:
                        if section['title']:
                            f.write(f"\n{section['title'].upper()}\n")
                        f.write(section['content'] + "\n\n")
                
                # Write references if available
                if result['references']:
                    f.write("REFERENCES:\n")
                    f.write("-" * 40 + "\n")
                    for i, ref in enumerate(result['references'], 1):
                        f.write(f"{i}. {ref}\n")
                    f.write("\n")
                
                # Write metadata
                if result['metadata']:
                    f.write("METADATA:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in result['metadata'].items():
                        if value:
                            f.write(f"{key}: {value}\n")
            
            logger.info(f"✅ Saved structured text to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save text file: {e}")
            return False
    
    def save_as_json(self, result: Dict[str, Any], output_path: str) -> bool:
        """Save conversion result as JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved structured JSON to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save JSON file: {e}")
            return False

def convert_single_pdf(pdf_path: str, output_dir: str = None, format: str = 'text') -> bool:
    """Convert a single PDF file"""
    try:
        converter = ProfessionalPDFConverter()
        
        # Convert PDF
        result = converter.convert_pdf(pdf_path)
        if not result:
            return False
        
        # Determine output path
        pdf_path = Path(pdf_path)
        if output_dir:
            output_path = Path(output_dir) / f"{pdf_path.stem}.{format}"
        else:
            output_path = pdf_path.with_suffix(f'.{format}')
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in requested format
        if format == 'json':
            return converter.save_as_json(result, str(output_path))
        else:
            return converter.save_as_text(result, str(output_path))
        
    except Exception as e:
        logger.error(f"❌ Failed to convert {pdf_path}: {e}")
        return False

def convert_pdf_directory(input_dir: str, output_dir: str = None, format: str = 'text') -> dict:
    """Convert all PDF files in a directory"""
    stats = {
        'files_found': 0,
        'files_converted': 0,
        'files_failed': 0
    }
    
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"❌ Input directory not found: {input_dir}")
            return stats
        
        # Find all PDF files
        pdf_files = list(input_path.glob('*.pdf')) + list(input_path.glob('*.PDF'))
        stats['files_found'] = len(pdf_files)
        
        logger.info(f"📁 Found {len(pdf_files)} PDF files in {input_dir}")
        
        for pdf_file in pdf_files:
            try:
                if convert_single_pdf(str(pdf_file), output_dir, format):
                    stats['files_converted'] += 1
                else:
                    stats['files_failed'] += 1
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                stats['files_failed'] += 1
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to process directory {input_dir}: {e}")
        return stats

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Professional PDF to Text Converter using GROBID and Unstructured',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single PDF file
  python professional_pdf_converter.py convert-file "path/to/paper.pdf"
  
  # Convert with custom output directory
  python professional_pdf_converter.py convert-file "path/to/paper.pdf" --output-dir "path/to/output"
  
  # Convert to JSON format
  python professional_pdf_converter.py convert-file "path/to/paper.pdf" --format json
  
  # Convert all PDF files in a directory
  python professional_pdf_converter.py convert-dir "path/to/pdf/directory"
  
  # Convert all PDFs with custom output directory
  python professional_pdf_converter.py convert-dir "path/to/pdf/directory" --output-dir "path/to/output"

Features:
  - Uses GROBID for high-quality academic document parsing
  - Falls back to Unstructured if GROBID unavailable
  - Extracts structured content (title, authors, abstract, sections)
  - Supports both text and JSON output formats
  - Handles references and metadata extraction

Requirements:
  - GROBID: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3
  - Unstructured: pip install 'unstructured[pdf]'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Convert single file command
    file_parser = subparsers.add_parser('convert-file', help='Convert a single PDF file')
    file_parser.add_argument('pdf_file', help='Path to PDF file to convert')
    file_parser.add_argument('--output-dir', help='Output directory for converted file')
    file_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    # Convert directory command
    dir_parser = subparsers.add_parser('convert-dir', help='Convert all PDF files in a directory')
    dir_parser.add_argument('directory', help='Path to directory containing PDF files')
    dir_parser.add_argument('--output-dir', help='Output directory for converted files')
    dir_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    logger.info("🚀 Starting Professional PDF Converter")
    logger.info("=" * 60)
    
    try:
        if args.command == 'convert-file':
            # Convert single PDF file
            success = convert_single_pdf(args.pdf_file, args.output_dir, args.format)
            if success:
                logger.info("✅ Conversion completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Conversion failed")
                sys.exit(1)
                
        elif args.command == 'convert-dir':
            # Convert all PDF files in directory
            stats = convert_pdf_directory(args.directory, args.output_dir, args.format)
            
            # Print summary
            logger.info(f"\n📊 CONVERSION SUMMARY:")
            logger.info(f"   📄 Files found: {stats['files_found']}")
            logger.info(f"   ✅ Files converted: {stats['files_converted']}")
            logger.info(f"   ❌ Files failed: {stats['files_failed']}")
            
            if stats['files_failed'] == 0:
                logger.info("✅ All conversions completed successfully")
                sys.exit(0)
            else:
                logger.warning(f"⚠️ {stats['files_failed']} files failed to convert")
                sys.exit(1)
        
        logger.info("\n💡 Next steps:")
        logger.info("   - Structured text files are ready for ingestion")
        logger.info("   - Use text_ingestion_pipeline.py to ingest the text files")
        logger.info("   - JSON files contain structured data for custom processing")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Converter failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
