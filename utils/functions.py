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

def evaluar_modelo_completo(modelo, X, y, model_name=""):
    """
    Evalúa completamente un modelo binario y retorna una lista de figuras.
    """
    figs = []
    
    # 1) Obtener probabilidades
    try:
        y_proba = modelo.predict_proba(X)[:, 1]
        # Verificar que y_proba tiene variedad
        unique_probs = np.unique(y_proba)
        print(f"[{model_name}] Valores únicos en y_proba: {unique_probs}")
        if unique_probs.size <= 1:
            st.warning(f"[{model_name}] `predict_proba` devuelve valores constantes. Las curvas ROC no se pueden generar.")
            y_proba = None
    except AttributeError:
        y_proba = None
        print(f"[{model_name}] AVISO: El modelo no soporta predict_proba(). Se usará predict() para la matriz de confusión.")
    
    # 2) Predicción con umbral=0.5
    if y_proba is not None:
        y_pred_05 = (y_proba >= 0.5).astype(int)
    else:
        y_pred_05 = modelo.predict(X)
    
    # 3) Matriz de confusión e informe (umbral=0.5)
    cm_05 = confusion_matrix(y, y_pred_05)
    print("\n=== [Umbral=0.5] Matriz de Confusión ===")
    print(cm_05)
    print("\n=== [Umbral=0.5] Classification Report ===")
    print(classification_report(y, y_pred_05, digits=4))
    
    # Figura de la Matriz de Confusión con umbral=0.5
    fig_cm_05, ax_cm_05 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_05, annot=True, fmt="d", cmap="Blues", ax=ax_cm_05, cbar=False)
    ax_cm_05.set_xlabel("Predicción")
    ax_cm_05.set_ylabel("Real")
    ax_cm_05.set_title(f"Matriz de Confusión (umbral=0.5) - {model_name}")
    figs.append(fig_cm_05)
    plt.close(fig_cm_05)
    
    # Si no hay predict_proba, retornar las figuras generadas hasta ahora
    if y_proba is None:
        return figs
    
    # 4) Encontrar el umbral que maximiza F1
    thresholds = np.linspace(0, 1, 101)
    precision_list, recall_list, f1_list = [], [], []
    
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        precision_list.append(precision_score(y, y_pred_thr, zero_division=0))
        recall_list.append(recall_score(y, y_pred_thr, zero_division=0))
        f1_list.append(f1_score(y, y_pred_thr, zero_division=0))
    
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)
    
    best_idx = np.argmax(f1_list)
    best_thr = thresholds[best_idx]
    best_f1 = f1_list[best_idx]
    
    # Predicción con el mejor umbral
    y_pred_best = (y_proba >= best_thr).astype(int)
    cm_best = confusion_matrix(y, y_pred_best)
    
    print(f"\n=== Mejor umbral para F1: {best_thr:.2f} (F1={best_f1:.4f}) ===")
    print("\n=== [Umbral que maximiza F1] Matriz de Confusión ===")
    print(cm_best)
    print("\n=== [Umbral que maximiza F1] Classification Report ===")
    print(classification_report(y, y_pred_best, digits=4))
    
    # Figura de la Matriz de Confusión con el mejor umbral
    fig_cm_best, ax_cm_best = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens", ax=ax_cm_best, cbar=False)
    ax_cm_best.set_xlabel("Predicción")
    ax_cm_best.set_ylabel("Real")
    ax_cm_best.set_title(f"Matriz de Confusión (umbral={best_thr:.2f}) - {model_name}")
    figs.append(fig_cm_best)
    plt.close(fig_cm_best)
    
    # Gráfico de Precisión, Recall y F1 Score vs umbral (Eliminar)
    # **Eliminado según solicitud**

    # 6) Curva ROC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax_roc.set_title('Curva ROC')
    ax_roc.legend(loc="lower right")
    figs.append(fig_roc)
    plt.close(fig_roc)
    
    # 7) Curva Precision-Recall (Eliminar)
    # **Eliminado según solicitud**

    # 8a) Recuento de TN, FP, FN, TP
    fig_counts, ax_counts = plt.subplots(figsize=(6, 4))
    counts = [cm_best[0, 0], cm_best[0, 1], cm_best[1, 0], cm_best[1, 1]]
    labels = ["TN", "FP", "FN", "TP"]
    colors = ["blue", "red", "orange", "green"]
    sns.barplot(x=labels, y=counts, palette=colors, ax=ax_counts)
    ax_counts.set_title("Recuento de TN, FP, FN, TP")
    ax_counts.set_xlabel("")
    ax_counts.set_ylabel("Cantidad")
    for i, v in enumerate(counts):
        ax_counts.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontsize=12)
    fig_counts.tight_layout()
    figs.append(fig_counts)
    plt.close(fig_counts)
    
    # 8b) Feature Importances
    if hasattr(modelo, "feature_importances_"):
        fi = modelo.feature_importances_
        top_k = 10
        top_indices = np.argsort(fi)[::-1][:top_k]
        
        # Verificar si X es un DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Asegurar que el número de features coincide
        if len(feature_names) != len(fi):
            st.error("El número de importancias de características no coincide con el número de columnas en X.")
        else:
            top_features = [feature_names[i] for i in top_indices]
            top_importances = fi[top_indices]
        
            fig_importances, ax_importances = plt.subplots(figsize=(10, 8))
            sns.barplot(x=top_importances, y=top_features, palette='viridis', ax=ax_importances)
            ax_importances.set_title("Top 10 Importancias de Características")
            ax_importances.set_xlabel("Importancia")
            ax_importances.set_ylabel("Características")
            
            # Añadir etiquetas con los valores de importancia
            for index, value in enumerate(top_importances):
                ax_importances.text(value + max(top_importances)*0.01, index, f"{value:.4f}", va='center', fontsize=10)
            
            fig_importances.tight_layout()
            figs.append(fig_importances)
            plt.close(fig_importances)
    else:
        st.warning("El modelo no tiene atributo 'feature_importances_'")
    
    return figs