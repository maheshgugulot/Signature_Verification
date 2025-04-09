import torch
import joblib
from feature_extraction import extract_features, lstm_model
from visualization import visualize_difference,relevance_score
from io import BytesIO
import base64
from PIL import Image

# Load models
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')



def predict(image_path):
    features = extract_features(image_path)

    if features is None:
        return {'error': 'Invalid image'}

    features = features.reshape(1, -1)

    # Predictions
    svm_raw = int(svm_model.predict(features)[0])
    knn_raw = int(knn_model.predict(features)[0])
    lstm_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        lstm_raw = int(torch.argmax(lstm_model(lstm_input), dim=1).item())

    predictions = {
        'SVM': svm_raw,
        'KNN': knn_raw,
        'LSTM': lstm_raw,
    }

    # Count votes
    counts = {0: 0, 1: 0}
    for pred in predictions.values():
        counts[pred] += 1

    # Additional result
    if counts[0] > counts[1]:
        vis_img = visualize_difference(image_path)
        if vis_img:
            buffered = BytesIO()
            vis_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            predictions['visualization'] = img_str
    elif counts[1] > counts[0]:
        sim_score = relevance_score(image_path)
        if sim_score is not None:
            predictions['similarity_score'] = float(sim_score)

    return predictions
