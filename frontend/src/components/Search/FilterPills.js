import React from 'react';
import styled from 'styled-components';

const pillLabels = {
  contentType: {
    abstract: 'Abstracts',
    methods: 'Methods',
    results: 'Results',
    discussion: 'Discussion',
    content: 'All Content',
  },
  timePeriod: {
    recent: 'Last 2 Years',
    '5year': 'Last 5 Years',
    decade: 'Last 10 Years',
    custom: 'Custom Range',
    all: 'All Years',
  },
  citationLevel: {
    high: 'High Impact',
    medium: 'Medium Impact',
    emerging: 'Emerging',
    all: 'All Papers',
  },
  dataQuality: {
    enriched: 'Complete Metadata',
    doi: 'Has DOI',
    all: 'All Papers',
  },
};

const defaultFilters = {
  contentType: 'content',
  timePeriod: 'all',
  citationLevel: 'all',
  dataQuality: 'all',
};

const PillsRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  margin-bottom: 1.2rem;
`;
const Pill = styled.span`
  display: inline-flex;
  align-items: center;
  background: linear-gradient(135deg, #007AFF 0%, #AF52DE 100%);
  color: #fff;
  border-radius: 16px;
  padding: 0.4rem 1.1rem 0.4rem 1rem;
  font-size: 0.97rem;
  font-weight: 500;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
`;
const RemoveBtn = styled.button`
  background: none;
  border: none;
  color: #fff;
  font-size: 1.1rem;
  margin-left: 0.5rem;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  opacity: 0.8;
  &:hover {
    opacity: 1;
  }
`;

export default function FilterPills({ filters, onRemove }) {
  const pills = Object.entries(filters)
    .filter(([cat, val]) => val && val !== defaultFilters[cat])
    .map(([cat, val]) => ({
      category: cat,
      label: pillLabels[cat]?.[val] || val,
    }));

  if (pills.length === 0) return null;

  return (
    <PillsRow>
      {pills.map(pill => (
        <Pill key={pill.category}>
          {pill.label}
          <RemoveBtn title="Remove filter" onClick={() => onRemove(pill.category)}>&times;</RemoveBtn>
        </Pill>
      ))}
    </PillsRow>
  );
} 