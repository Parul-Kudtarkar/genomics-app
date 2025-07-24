import re
from typing import List, Dict

class SectionAwareChunker:
    def __init__(self):
        self.section_patterns = {
            'abstract': r'abstract\s*:?.{0,20}\n',
            'introduction': r'introduction\s*:?.{0,20}\n',
            'methods': r'(?:methods|methodology|materials\s+and\s+methods)\s*:?.{0,20}\n',
            'results': r'results\s*:?.{0,20}\n',
            'discussion': r'discussion\s*:?.{0,20}\n',
            'conclusion': r'conclusion\s*:?.{0,20}\n',
            'references': r'(?:references|bibliography)\s*:?.{0,20}\n'
        }
        self.section_order = [
            'abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'references'
        ]

    def _detect_sections(self, text: str) -> List[Dict]:
        """Detects sections in the text and returns a list of dicts with type and content."""
        lower_text = text.lower()
        section_indices = []
        for section, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, lower_text):
                section_indices.append((match.start(), section))
        section_indices.sort()
        sections = []
        for i, (start, section) in enumerate(section_indices):
            end = section_indices[i+1][0] if i+1 < len(section_indices) else len(text)
            content = text[start:end].strip()
            sections.append({'type': section, 'text': content})
        if not sections:
            # fallback: treat all as one section
            sections = [{'type': 'content', 'text': text.strip()}]
        return sections

    def _chunk_with_context_preservation(self, section: Dict, chunk_size: int = 8000, overlap: int = 200) -> List[Dict]:
        text = section['text']
        doc_id = section.get('doc_id', '')
        title = section.get('title', '')
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'id': f"{doc_id}_{section['type']}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'doc_id': doc_id,
                    'title': title,
                    'section_type': section['type']
                })
                chunk_index += 1
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def create_intelligent_chunks(self, text: str, doc_id: str, title: str) -> List[Dict]:
        sections = self._detect_sections(text)
        chunks = []
        for section in sections:
            if section['type'] in ['references', 'bibliography']:
                break
            section['doc_id'] = doc_id
            section['title'] = title
            # For now, use context preservation for all
            chunks.extend(self._chunk_with_context_preservation(section))
        return chunks 