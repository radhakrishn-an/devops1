# Docker Setup

## 1. Build the Docker Image

```bash
docker build -t radhakrish55/boston-housing-api:latest .
```

---

## 2. Run the Docker Container

```bash
docker run -p 8000:8000 radhakrish55/boston-housing-api:latest
```

The API will now be available at

```text
http://localhost:8000
```

---

## 3. Push the Image to Docker Hub

Login to Docker Hub

```bash
docker login
```

Push the image

```bash
docker push radhakrish55/boston-housing-api:latest
```

---

## 4. Pull the Image

Anyone can pull the image using

```bash
docker pull radhakrish55/boston-housing-api:latest
```

---

## 5. Run the Pulled Image

```bash
docker run -p 8000:8000 radhakrish55/boston-housing-api:latest
```

---

# API Endpoints

## Home

**GET**

```text
/
```

---

## Health Check

**GET**

```text
/health
```

### Response

```json
{
    "status": "ok"
}
```

---

## Prediction

**POST**

```text
/predict
```

### Request

```json
{
    "features": [
        0.00632,
        18.0,
        2.31,
        0,
        0.538,
        6.575,
        65.2,
        4.09,
        1,
        296,
        15.3,
        396.9,
        4.98
    ]
}
```

### Example Response

```json
{
    "predicted_price": 24.53
}
```

---

# Docker Hub Repository

```text
https://hub.docker.com/r/radhakrish55/boston-housing-api
```

---
