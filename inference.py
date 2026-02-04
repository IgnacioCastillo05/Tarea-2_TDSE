"""
Script de inference para SageMaker endpoint
"""
import json
import numpy as np
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    """Carga el modelo desde el directorio."""
    model_path = os.path.join(model_dir, 'model.json')
    with open(model_path, 'r') as f:
        model_params = json.load(f)
    return model_params

def input_fn(request_body, content_type='application/json'):
    """Procesa el input del usuario."""
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_params):
    """Hace la predicción."""
    feature_names = model_params['feature_names']
    X = np.array([input_data[feat] for feat in feature_names])
    
    mean = np.array(model_params['feature_mean'])
    std = np.array(model_params['feature_std'])
    X_norm = (X - mean) / std
    
    w = np.array(model_params['weights'])
    b = model_params['bias']
    
    z = np.dot(w, X_norm) + b
    probability = float(sigmoid(z))
    prediction = 1 if probability >= 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'diagnosis': 'Presence' if prediction == 1 else 'Absence',
        'confidence': probability if prediction == 1 else (1 - probability)
    }

def output_fn(prediction, accept='application/json'):
    """Formatea la respuesta."""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")