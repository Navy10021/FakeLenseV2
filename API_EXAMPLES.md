# FakeLenseV2 API Examples

This document provides comprehensive examples for using the FakeLenseV2 REST API.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Endpoints](#endpoints)
5. [Request/Response Examples](#requestresponse-examples)
6. [Error Handling](#error-handling)
7. [Code Examples](#code-examples)

---

## Getting Started

### Starting the API Server

```bash
# Option 1: Direct Python
python -m code.api_server

# Option 2: Using Docker
docker-compose up fakelense-api

# Option 3: Using uvicorn directly
uvicorn code.api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:
- API key authentication
- OAuth 2.0
- JWT tokens

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

| Endpoint | Rate Limit |
|----------|-----------|
| `/predict` | 100 requests/minute per IP |
| `/batch_predict` | 20 requests/minute per IP |
| Other endpoints | 200 requests/minute per IP |

When rate limit is exceeded, you'll receive a `429 Too Many Requests` response.

---

## Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API status and version information.

### 2. Health Check
**GET** `/health`

Check if the API and model are healthy.

### 3. Single Prediction
**POST** `/predict`

Predict if a single news article is fake, suspicious, or real.

### 4. Batch Prediction
**POST** `/batch_predict`

Predict multiple articles in a single request.

---

## Request/Response Examples

### 1. Root Endpoint

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "version": "2.0.0",
  "uptime_seconds": 3600.5
}
```

---

### 2. Health Check

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0",
  "uptime_seconds": 3600.5
}
```

---

### 3. Single Prediction - Fake News

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking: Aliens have landed in New York City and are demanding to speak to world leaders. Unconfirmed reports suggest they brought advanced technology.",
    "source": "Unknown Blog",
    "social_reactions": 50000
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Fake News",
  "confidence": 0.92,
  "all_probabilities": {
    "fake": 0.92,
    "suspicious": 0.06,
    "real": 0.02
  },
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Response Headers:**
```
X-Request-ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

### 4. Single Prediction - Real News

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Federal Reserve announced today that it will maintain interest rates at current levels, citing stable economic indicators and moderate inflation. This decision was made after a two-day meeting of the Federal Open Market Committee.",
    "source": "Reuters",
    "social_reactions": 10000
  }'
```

**Response:**
```json
{
  "prediction": 2,
  "label": "Real News",
  "confidence": 0.95,
  "all_probabilities": {
    "fake": 0.02,
    "suspicious": 0.03,
    "real": 0.95
  },
  "request_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901"
}
```

---

### 5. Single Prediction - Suspicious News

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "According to anonymous sources, a major tech company is planning to release a revolutionary product next week that will change everything. Details are scarce but insiders claim it will be huge.",
    "source": "Tech Blog",
    "social_reactions": 5000
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Suspicious News",
  "confidence": 0.73,
  "all_probabilities": {
    "fake": 0.15,
    "suspicious": 0.73,
    "real": 0.12
  },
  "request_id": "c3d4e5f6-a7b8-9012-cdef-123456789012"
}
```

---

### 6. Minimal Request (Optional Fields Omitted)

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test article with minimal fields provided. It has enough length to pass validation."
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Suspicious News",
  "confidence": 0.65,
  "all_probabilities": {
    "fake": 0.25,
    "suspicious": 0.65,
    "real": 0.10
  },
  "request_id": "d4e5f6a7-b8c9-0123-def1-234567890123"
}
```

---

### 7. Batch Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {
        "text": "Breaking news: Major scientific breakthrough announced by researchers at leading university. The discovery could revolutionize medicine.",
        "source": "Science News",
        "social_reactions": 15000
      },
      {
        "text": "Celebrity spotted eating at local restaurant. Sources say they ordered the special.",
        "source": "Gossip Blog",
        "social_reactions": 30000
      },
      {
        "text": "Stock market closes higher today following positive economic data. Dow Jones gained 150 points.",
        "source": "Bloomberg",
        "social_reactions": 8000
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "label": "Suspicious News",
      "confidence": 0.68,
      "all_probabilities": {
        "fake": 0.18,
        "suspicious": 0.68,
        "real": 0.14
      }
    },
    {
      "prediction": 0,
      "label": "Fake News",
      "confidence": 0.85,
      "all_probabilities": {
        "fake": 0.85,
        "suspicious": 0.10,
        "real": 0.05
      }
    },
    {
      "prediction": 2,
      "label": "Real News",
      "confidence": 0.93,
      "all_probabilities": {
        "fake": 0.03,
        "suspicious": 0.04,
        "real": 0.93
      }
    }
  ],
  "request_id": "e5f6a7b8-c9d0-1234-ef12-345678901234",
  "total": 3,
  "success": 3,
  "errors": 0
}
```

---

## Error Handling

### Validation Error (422)

**Request with text too short:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Short"
  }'
```

**Response:**
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 10 characters",
      "input": "Short",
      "ctx": {
        "min_length": 10
      }
    }
  ]
}
```

---

### Model Not Loaded (503)

**Response:**
```json
{
  "detail": "Model not loaded"
}
```

---

### Rate Limit Exceeded (429)

**Response:**
```json
{
  "error": "Rate limit exceeded: 100 per 1 minute"
}
```

---

### Server Error (500)

**Response:**
```json
{
  "detail": "Prediction failed: Internal server error"
}
```

---

## Code Examples

### Python with requests

```python
import requests

# Single prediction
def predict_article(text, source=None, social_reactions=0):
    url = "http://localhost:8000/predict"
    payload = {
        "text": text,
        "source": source,
        "social_reactions": social_reactions
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['all_probabilities']}")
        print(f"Request ID: {result['request_id']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Example usage
article_text = """
The Federal Reserve announced today a decision to maintain
interest rates at current levels. This comes after careful
analysis of economic indicators and inflation data.
"""

result = predict_article(article_text, source="Reuters", social_reactions=5000)
```

---

### Python with httpx (async)

```python
import httpx
import asyncio

async def predict_article_async(text, source=None, social_reactions=0):
    url = "http://localhost:8000/predict"
    payload = {
        "text": text,
        "source": source,
        "social_reactions": social_reactions
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        return response.json()

# Example usage
async def main():
    article_text = "Breaking news about technology..."
    result = await predict_article_async(article_text, source="TechCrunch")
    print(result)

asyncio.run(main())
```

---

### JavaScript/Node.js with fetch

```javascript
async function predictArticle(text, source = null, socialReactions = 0) {
  const url = 'http://localhost:8000/predict';

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      source: source,
      social_reactions: socialReactions
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  console.log(`Prediction: ${data.label}`);
  console.log(`Confidence: ${(data.confidence * 100).toFixed(2)}%`);
  console.log(`Request ID: ${data.request_id}`);

  return data;
}

// Example usage
const article = "The stock market reached new highs today...";
predictArticle(article, "Bloomberg", 10000)
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

---

### JavaScript/TypeScript with axios

```typescript
import axios from 'axios';

interface PredictionRequest {
  text: string;
  source?: string;
  social_reactions?: number;
}

interface PredictionResponse {
  prediction: number;
  label: string;
  confidence: number;
  all_probabilities: {
    fake: number;
    suspicious: number;
    real: number;
  };
  request_id: string;
}

async function predictArticle(
  request: PredictionRequest
): Promise<PredictionResponse> {
  try {
    const response = await axios.post<PredictionResponse>(
      'http://localhost:8000/predict',
      request
    );

    console.log(`Prediction: ${response.data.label}`);
    console.log(`Confidence: ${(response.data.confidence * 100).toFixed(2)}%`);

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('API Error:', error.response?.data);
    }
    throw error;
  }
}

// Example usage
const article: PredictionRequest = {
  text: "Breaking news about a major event...",
  source: "News Source",
  social_reactions: 5000
};

predictArticle(article);
```

---

### cURL with Pretty Output

```bash
# Single prediction with formatted output
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your article text here...",
    "source": "Reuters",
    "social_reactions": 1000
  }' | jq '.'

# Extract specific fields
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your article text here..."
  }' | jq '.label, .confidence'

# Save request ID to variable
REQUEST_ID=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Article text..."}' | jq -r '.request_id')

echo "Request ID: $REQUEST_ID"
```

---

### Batch Processing Example

```python
import requests
from typing import List, Dict

def predict_batch(articles: List[Dict[str, any]]) -> Dict:
    """
    Predict multiple articles in batches of 100.

    Args:
        articles: List of article dictionaries with 'text', 'source', and 'social_reactions'

    Returns:
        Combined results from all batches
    """
    url = "http://localhost:8000/batch_predict"
    batch_size = 100
    all_results = []

    # Process in batches
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        payload = {"articles": batch}

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            all_results.extend(result['predictions'])
            print(f"Processed batch {i//batch_size + 1}: {result['success']}/{result['total']} succeeded")
        else:
            print(f"Error in batch {i//batch_size + 1}: {response.status_code}")

    return {
        "total_processed": len(all_results),
        "predictions": all_results
    }

# Example usage
articles = [
    {"text": "Article 1 text...", "source": "Source1", "social_reactions": 100},
    {"text": "Article 2 text...", "source": "Source2", "social_reactions": 200},
    # ... more articles
]

results = predict_batch(articles)
print(f"Total articles processed: {results['total_processed']}")
```

---

## Response Field Descriptions

### Prediction Response

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | integer | Predicted class: 0 = Fake, 1 = Suspicious, 2 = Real |
| `label` | string | Human-readable label: "Fake News", "Suspicious News", or "Real News" |
| `confidence` | float | Confidence score for the prediction (0.0 to 1.0) |
| `all_probabilities` | object | Probabilities for all three classes |
| `all_probabilities.fake` | float | Probability of being fake news (0.0 to 1.0) |
| `all_probabilities.suspicious` | float | Probability of being suspicious news (0.0 to 1.0) |
| `all_probabilities.real` | float | Probability of being real news (0.0 to 1.0) |
| `request_id` | string | Unique identifier for tracking this request (UUID format) |

---

## Best Practices

1. **Always check the confidence score**: Lower confidence may indicate uncertainty
2. **Use batch prediction for multiple articles**: More efficient than individual requests
3. **Handle rate limiting**: Implement exponential backoff when receiving 429 responses
4. **Store request IDs**: Useful for debugging and tracking
5. **Validate input**: Ensure text is at least 10 characters long
6. **Monitor response times**: Typical response times are 50-200ms per article

---

## Troubleshooting

### High Latency
- Check if model is loaded (`/health` endpoint)
- Consider using batch prediction
- Ensure adequate server resources

### Low Confidence Scores
- Article may be ambiguous
- Consider the context and source reliability
- Check if the article is in a domain the model was trained on

### Rate Limiting
- Implement request queuing
- Use batch prediction
- Contact admin for higher limits if needed

---

## Support

For issues or questions:
- **GitHub**: https://github.com/Navy10021/FakeLenseV2/issues
- **Email**: iyunseob4@gmail.com

---

**Version**: 2.0.0
**Last Updated**: 2025-11-06
