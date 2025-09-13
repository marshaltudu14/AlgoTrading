# API Documentation

## Overview

The Transformer Trading System provides a RESTful API for:

- Model training and inference
- Trading signal generation
- Data management
- Performance monitoring

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

API endpoints require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### Model Management
- `POST /models/train` - Train a new model
- `GET /models/{model_id}` - Get model information
- `DELETE /models/{model_id}` - Delete a model

### Prediction
- `POST /predict` - Generate trading predictions
- `POST /predict/batch` - Generate batch predictions

### Data
- `GET /data/symbols` - Get available trading symbols
- `GET /data/historical/{symbol}` - Get historical data
- `POST /data/upload` - Upload data files

### Monitoring
- `GET /monitoring/performance` - Get performance metrics
- `GET /monitoring/logs` - Get system logs

## Error Responses

All endpoints return standardized error responses:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {}
  }
}
```

## Rate Limiting

- 100 requests per minute
- 1000 requests per hour