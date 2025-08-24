# Challenge Documentation

## Part I: Model Implementation

### Model Selection

After analyzing the notebook results, I chose **XGBoost with class balancing and top 10 features** for the following reasons:

1. **Performance**: Both XGBoost and Logistic Regression showed similar performance, but XGBoost typically handles non-linear relationships better and is more robust to outliers.

2. **Class Balancing**: The dataset is highly imbalanced (~82% no delay, ~18% delay). Using `scale_pos_weight` in XGBoost improves recall for the minority class (delays), which is critical for this use case.

3. **Feature Selection**: Using only the top 10 features:
   - Reduces model complexity and inference time
   - Maintains similar performance to using all features
   - Makes the API more efficient

4. **Business Impact**: In flight delay prediction, false negatives (missing actual delays) are more costly than false positives. The balanced model prioritizes recall.

### Bugs Fixed

1. **`get_rate_from_column` function**: The rate calculation was inverted. It was calculating `total / delays` instead of `delays / total * 100`.

2. **`get_period_day` function**: Added edge case handling for times exactly at boundaries (e.g., 5:00 AM).

3. **Character encoding**: Removed special characters (ñ, í) that could cause encoding issues in production.

4. **Data preprocessing**: Added validation to ensure all required features are present before prediction.

### Good Practices Applied

1. **Type hints**: Added proper type annotations for better code clarity and IDE support.

2. **Docstrings**: Comprehensive documentation for all methods.

3. **Error handling**: Added validation for untrained models.

4. **Separation of concerns**: Clearly separated preprocessing, training, and prediction logic.

5. **Consistent naming**: Used PEP 8 naming conventions.

6. **Immutable top features**: Stored as a class attribute to ensure consistency.

7. **Module structure**: Following the provided template structure with challenge package.

## Part II: API Implementation

The API was implemented using FastAPI with the following endpoints:

- `GET /health`: Health check endpoint
- `POST /predict`: Prediction endpoint accepting flight data

Key features:
- Input validation using Pydantic models
- Proper error handling with appropriate HTTP status codes
- JSON response format for easy integration

## Part III: Cloud Deployment

Deployed on Google Cloud Platform using:
- Cloud Run for serverless container hosting
- Docker for containerization
- Artifact Registry for container storage

Advantages:
- Auto-scaling based on traffic
- Pay-per-use pricing model
- Built-in HTTPS and load balancing

## Part IV: CI/CD Implementation

### CI Pipeline (`ci.yml`)
- Runs on every push and pull request
- Steps:
  1. Checkout code
  2. Set up Python environment
  3. Install dependencies
  4. Run model tests
  5. Run API tests

### CD Pipeline (`cd.yml`)
- Triggers on push to main branch
- Steps:
  1. Build Docker image
  2. Push to GCP Artifact Registry
  3. Deploy to Cloud Run
  4. Run stress tests

