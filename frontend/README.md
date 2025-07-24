# Diabetes Research Assistant Frontend

This is a modern React frontend for the Diabetes Research Assistant, featuring Apple Intelligence design, advanced metadata handling, and robust research UI components.

## Features
- Apple-style design with gradients and smooth animations
- Advanced search with filter panel and suggestions
- Rich result cards with Crossref-preferred metadata
- Data visualizations (timeline, citations, journals, quality)
- Smart research features (related papers, trends, author explorer)
- Handles incomplete metadata gracefully

## Getting Started

1. **Install dependencies:**
   ```bash
   npm install
   ```
2. **Start the development server:**
   ```bash
   npm start
   ```
   The app will run at http://localhost:3000 and proxy API requests to http://localhost:8000.

## Project Structure
```
src/
  components/
    Search/
    Results/
    Visualizations/
    Smart/
  utils/
  hooks/
  App.js
  index.js
```

## API
- Expects a backend at `/api/query` with the response format described in the system guide.

## Customization
- Add or modify components in `src/components` for new features or UI tweaks.
- Utility functions for metadata are in `src/utils/metadataHelpers.js`.

---

For more, see the system implementation guide in the project root. 