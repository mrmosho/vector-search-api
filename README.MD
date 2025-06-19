# Vector Search API

A FastAPI-based web service for semantic vector search using FAISS and sentence transformers. Converts your CLI vector search engine into a deployable web API.

## Features

- **FastAPI** with automatic API documentation
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for text embeddings
- **Automatic index building** from CSV data
- **RESTful API** with JSON responses
- **Ready for Render deployment**

## Quick Start

### 1. Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your data:**
   - Place your `data.csv` file in the project root
   - Ensure it has a `DESCRIPTION` column (or modify the code for your column name)

3. **Start the server:**
```bash
uvicorn main:app --reload
```

4. **Test the API:**
   - Open http://localhost:8000/docs for interactive API documentation
   - Or run the test script: `python test_api.py`

### 2. API Endpoints

#### GET `/health`
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "search_engine_initialized": true,
  "total_vectors": 1000
}
```

#### GET `/search?q=your_query&k=5`
Search for similar vectors
```json
{
  "method": "get",
  "status": "success",
  "text": "your search query",
  "results": [
    {
      "euclidean_distance": 0.1234,
      "data": {
        "ID": 1,
        "TITLE": "Example Title",
        "DESCRIPTION": "Example description...",
        "SOURCE_NAME": "Source",
        "LANGUAGE": "EN"
      }
    }
  ]
}
```

#### GET `/rebuild-index`
Rebuild the search index from updated data.csv

### 3. Deploy to Render

1. **Connect your GitHub repo to Render**

2. **Use these settings:**
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables:**
   - `PYTHON_VERSION`: 3.11.0
   - `PYTHONUNBUFFERED`: 1

4. **Upload your data.csv** to the deployment or use file upload via API

## File Structure

```
vector-search-api/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── render.yaml         # Render deployment config
├── test_api.py         # API testing script
├── data.csv            # Your data (not in repo)
├── index.ivf           # FAISS index (auto-generated)
├── metadata.json       # Metadata cache (auto-generated)
└── README.md           # This file
```

## API Response Format

As agreed with your team, all search responses follow this format:

```json
{
  "method": "get",
  "status": "success" | "error",
  "text": "user query text",
  "results": [
    {
      "euclidean_distance": 0.1234,
      "data": { ...metadata... }
    }
  ],
  "error": "error message if status is error"
}
```

## Development Notes

- **First run:** The API will automatically build the FAISS index from `data.csv`
- **Index persistence:** The index is saved to disk and reloaded on restart
- **Text column:** Currently expects `DESCRIPTION` column - modify in `main.py` if different
- **Model:** Uses `all-MiniLM-L6-v2` sentence transformer model
- **Auto-scaling:** IVF clusters are automatically determined based on dataset size

## Testing

1. **Start the server:**
```bash
uvicorn main:app --reload
```

2. **Run tests:**
```bash
python test_api.py
```

3. **Manual testing:**
   - Visit http://localhost:8000/docs
   - Use the interactive Swagger UI to test endpoints

## Troubleshooting

**"data.csv not found":**
- Ensure `data.csv` is in the project root directory
- Check that the file has the expected columns

**"Search engine not initialized":**
- Check the server logs for initialization errors
- Verify your CSV data format and text column name

**Memory issues:**
- Large datasets may require more RAM
- Consider using smaller batch sizes for embedding generation

**Model download issues:**
- First run downloads the sentence transformer model
- Ensure internet connection for initial setup

## Next Steps

- Your React frontend team can now integrate with these API endpoints
- The `/docs` endpoint provides complete API documentation
- Consider adding authentication if needed for production
- Monitor performance and adjust FAISS parameters as needed