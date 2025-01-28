from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score
import pandas as pd

SEED = 42

def calcular_metricas_adicionales(modelo, X, y, y_pred):
    """Calcula ROC-AUC y Precision-Recall además de Accuracy y F1."""
    roc_auc = roc_auc_score(y, modelo.predict_proba(X)[:, 1]) if hasattr(modelo, "predict_proba") else None
    precision, recall, _ = precision_recall_curve(y, modelo.predict_proba(X)[:, 1]) if hasattr(modelo, "predict_proba") else (None, None, None)
    return {
        "F1": f1_score(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "ROC-AUC": roc_auc,
        "Precision-Recall": {"Precision": precision, "Recall": recall} if precision is not None else None
    }


def probar_modelo_con_grid(modelo, X_train, y_train, X_validation, y_validation, param_grid):
    """
    Entrena el modelo con GridSearchCV y evalúa F1/accuracy en entrenamiento y validación.

    Args:
        modelo: Instancia del estimador de scikit-learn.
        X_train, y_train: Datos de entrenamiento.
        X_validation, y_validation: Datos de validación.
        param_grid: Diccionario con parámetros para GridSearchCV.

    Returns:
        Tupla con (F1 validation, Accuracy validation, mejores parámetros, mejor modelo, F1 train).
    """
    if X_train.empty or X_validation.empty:
        raise ValueError("X_train/X_validation no pueden estar vacíos.")

    # --- Configuración universal de random_state ---
    if hasattr(modelo, 'random_state'):
        modelo.set_params(random_state=SEED)
    if hasattr(modelo, 'n_jobs'):
        modelo.set_params(n_jobs=1)  # Prevenir conflictos de paralelismo

    # --- Validar parámetros del modelo ---
    valid_params = [param for param in param_grid.keys() if param in modelo.get_params().keys()]
    param_grid = {k: v for k, v in param_grid.items() if k in valid_params}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        modelo, param_grid, cv=cv, scoring='f1',
        n_jobs=2, verbose=2, error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred_train = best_model.predict(X_train)
    y_pred_validation = best_model.predict(X_validation)

    # Métricas adicionales
    metricas_train = calcular_metricas_adicionales(best_model, X_train, y_train, y_pred_train)
    metricas_validation = calcular_metricas_adicionales(best_model, X_validation, y_validation, y_pred_validation)

    return {
        "Train": metricas_train,
        "Validation": metricas_validation,
        "Best Params": best_params,
        "Best Model": best_model
    }

from sklearn.base import BaseEstimator

def probar_modelo_con_random_search(modelo, X_train, y_train, X_validation, y_validation, param_distributions, n_iter=50):
    """
    Entrena el modelo con RandomizedSearchCV y evalúa métricas.
    """
    # Clonar modelo para evitar efectos secundarios
    modelo_clonado = clone(modelo)

    # Configurar parámetros de ajuste para XGBoost
    fit_params = {
        "eval_set": [(X_validation, y_validation)],  # Proporcionar conjunto de validación
        "early_stopping_rounds": 10,                # Configurar early stopping
        "verbose": False                            # Evitar exceso de logs
    } if isinstance(modelo_clonado, BaseEstimator) and hasattr(modelo_clonado, "fit") and "eval_set" in modelo_clonado.fit.__code__.co_varnames else {}

    # Configurar RandomizedSearchCV
    random_search = RandomizedSearchCV(
        modelo_clonado,
        param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring='f1',
        n_jobs=-1,
        random_state=SEED,
        verbose=2
    )

    # Ajustar el modelo usando `fit_params` (solo aplica a XGBoost y otros modelos compatibles)
    random_search.fit(X_train, y_train, **fit_params)

    # Obtener el mejor modelo y parámetros
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Evaluar
    y_pred_train = best_model.predict(X_train)
    y_pred_validation = best_model.predict(X_validation)

    # Métricas adicionales
    metricas_train = calcular_metricas_adicionales(best_model, X_train, y_train, y_pred_train)
    metricas_validation = calcular_metricas_adicionales(best_model, X_validation, y_validation, y_pred_validation)

    return {
        "Train": metricas_train,
        "Validation": metricas_validation,
        "Best Params": best_params,
        "Best Model": best_model
    }



def graficar_precision_recall(precision, recall, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.title(f"Curva Precision-Recall para {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.show()

def configure_eval_set(model_name, X_train, y_train, X_validation, y_validation):
    """
    Configura eval_set para modelos que lo requieran, como XGBoost, y otros que soporten validación interna.

    Args:
        model_name (str): Nombre del modelo actual (p. ej., "XGBoost").
        X_train, y_train: Datos de entrenamiento.
        X_validation, y_validation: Datos de validación.

    Returns:
        dict: Diccionario con parámetros adicionales para el modelo.
    """
    eval_set = [(X_train, y_train), (X_validation, y_validation)]  # Usar tanto train como validation para métricas internas

    if "XGBoost" in model_name:  # Si es un modelo de XGBoost
        return {
            "eval_set": eval_set,
            "eval_metric": ["logloss", "auc"],  # Añadir ROC-AUC y logloss como métricas internas
            "early_stopping_rounds": 10,       # Evitar sobreajuste
            "verbose": False                   # Desactivar logs
        }
    # Para otros modelos que no necesiten eval_set
    return {}

def imprimir_resultados(resultado):
    print("\n=== Resultados del Modelo ===")
    print("Mejores Parámetros:")
    print(resultado["Best Params"])
    print("\nMétricas de Entrenamiento:")
    for k, v in resultado["Train"].items():
        if isinstance(v, dict):  # Para Precision-Recall
            print(f"  {k}: [Gráfico disponible]")
        else:
            print(f"  {k}: {v:.4f}")
    print("\nMétricas de Validación:")
    for k, v in resultado["Validation"].items():
        if isinstance(v, dict):  # Para Precision-Recall
            print(f"  {k}: [Gráfico disponible]")
        else:
            print(f"  {k}: {v:.4f}")

def evaluar_en_test(modelo, X_test, y_test):
    y_pred_test = modelo.predict(X_test)
    metricas_test = calcular_metricas_adicionales(modelo, X_test, y_test, y_pred_test)
    print("\nMétricas en Test:")
    for k, v in metricas_test.items():
        if isinstance(v, dict):  # Para Precision-Recall
            print(f"  {k}: [Gráfico disponible]")
        else:
            print(f"  {k}: {v:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
)
import numpy as np
import streamlit as st

def evaluar_modelo_completo(model, X_test, y_test, model_name="Modelo"):
    """
    Evalúa un modelo de clasificación y genera varias visualizaciones de rendimiento.

    Args:
        model: Modelo de clasificación ya entrenado.
        X_test (pd.DataFrame o np.ndarray): Características de prueba.
        y_test (array-like): Etiquetas verdaderas de prueba.
        model_name (str): Nombre del modelo para etiquetas en las gráficas.

    Returns:
        list: Lista de figuras de matplotlib generadas.
    """
    figuras = []

    # Predicciones y probabilidades
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None
    y_pred = model.predict(X_test)

    # 1. Informe de Clasificación
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    # Filtramos para que solo se muestren las métricas más relevantes (evitamos 'accuracy' duplicado)
    filtered_report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

    fig_report, ax_report = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        filtered_report_df.iloc[:, :-1],  # Solo mostramos columnas de métricas relevantes
        annot=True,
        fmt=".3f",  # Más decimales para mejor análisis
        cmap="coolwarm",  # Más contraste en los colores
        linewidths=0.5,  # Líneas separadoras más finas
        cbar=True,  # Agregar barra de colores
        ax=ax_report
    )
    ax_report.set_title(f"Informe de Clasificación - {model_name}")
    ax_report.set_xlabel("Métrica")
    ax_report.set_ylabel("Clase")
    figuras.append(fig_report)

    # 3. Curva ROC
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        sns.lineplot(x=fpr, y=tpr, label=f'AUC = {roc_auc:.2f}', ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
        ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax_roc.set_title(f"Curva ROC - {model_name}")
        ax_roc.legend(loc='lower right')
        figuras.append(fig_roc)

    # 5. Importancia de Características (Solo para modelos que lo soportan, como XGBoost)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X_test.columns if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(20)  # Top 20

        fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_imp, palette='viridis')
        ax_imp.set_title(f"Importancia de Características - {model_name}")
        ax_imp.set_xlabel("Importancia")
        ax_imp.set_ylabel("Características")
        figuras.append(fig_imp)

    # 6. Distribución de Probabilidades por Clase
    if y_proba is not None:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
        sns.kdeplot(y_proba[y_test == 1], label="Clase 1", shade=True, ax=ax_dist)
        sns.kdeplot(y_proba[y_test == 0], label="Clase 0", shade=True, ax=ax_dist)
        ax_dist.set_xlabel("Probabilidad Predicha")
        ax_dist.set_ylabel("Densidad")
        ax_dist.set_title(f"Distribución de Probabilidades por Clase - {model_name}")
        ax_dist.legend()
        figuras.append(fig_dist)

    return figuras