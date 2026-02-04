"""
Script de entrenamiento para SageMaker
"""
import numpy as np
import pandas as pd
import json
import os
import argparse

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(w, b, X, y):
    m = X.shape[0]
    z = X @ w + b
    f = sigmoid(z)
    eps = 1e-8
    f_clipped = np.clip(f, eps, 1 - eps)
    J = -(1 / m) * np.sum(y * np.log(f_clipped) + (1 - y) * np.log(1 - f_clipped))
    return J

def compute_gradient(w, b, X, y):
    m = X.shape[0]
    z = X @ w + b
    f = sigmoid(z)
    error = f - y
    dj_dw = (1 / m) * (X.T @ error)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w = w_init.copy()
    b = b_init
    J_history = []
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(w, b, X, y)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J = compute_cost(w, b, X, y)
        J_history.append(J)
        
        if i % 500 == 0:
            print(f"Iter {i}: Cost = {J:.6f}")
    
    return w, b, J_history

def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--num_iters', type=int, default=1500)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args = parser.parse_args()
    
    print("Cargando datos...")
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))

    feature_cols = ['Age', 'Cholesterol', 'BP', 'Max HR', 'ST depression', 
                    'Number of vessels fluro', 'Chest pain type', 'Thallium']
    
    X_train = train_data[feature_cols].values
    y_train = train_data['Heart Disease'].values

    X_train_norm, _, mean, std = normalize(X_train, X_train)
    
    print(f"Entrenando modelo con alpha={args.alpha}, iters={args.num_iters}")

    n_features = X_train_norm.shape[1]
    w_init = np.zeros(n_features)
    b_init = 0.0
    
    w_final, b_final, J_history = gradient_descent(
        X_train_norm, y_train, w_init, b_init, args.alpha, args.num_iters
    )
    
    print(f"Entrenamiento completado. Costo final: {J_history[-1]:.6f}")

    model_params = {
        'weights': w_final.tolist(),
        'bias': float(b_final),
        'feature_names': feature_cols,
        'feature_mean': mean.tolist(),
        'feature_std': std.tolist(),
        'alpha': args.alpha,
        'num_iters': args.num_iters,
        'final_cost': float(J_history[-1])
    }
    
    model_path = os.path.join(args.model_dir, 'model.json')
    with open(model_path, 'w') as f:
        json.dump(model_params, f)
    
    print(f"Modelo guardado en: {model_path}")