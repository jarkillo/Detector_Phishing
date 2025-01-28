import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin

# --- Clase para eliminar columnas ---
class DropColumns(BaseEstimator, TransformerMixin):
    """
    Elimina columnas específicas de un DataFrame, manejando casos donde las columnas no existen.

    Args:
        columns (list): Lista de nombres de columnas a eliminar.
    """

    def __init__(self, columns):
        self.columns = columns
        self.cols_found_ = []  # Columnas encontradas durante el fit

    def fit(self, X, y=None):
        """Identifica las columnas presentes en el dataset."""
        self.cols_found_ = [col for col in self.columns if col in X.columns]
        if not self.cols_found_:
            warnings.warn("No se encontraron columnas para eliminar.", UserWarning)
        return self

    def transform(self, X):
        """Elimina las columnas identificadas."""
        X_transformed = X.drop(columns=self.cols_found_, errors='ignore')
        # Verificar si se eliminaron efectivamente
        cols_removed = list(set(self.cols_found_) - set(X_transformed.columns))
        if cols_removed:
            print(f"Columnas eliminadas: {cols_removed}")
        return X_transformed


# =============================================================================
#               Funciones auxiliares para permitir pickle (sin lambdas)
# =============================================================================

def page_rank_condition(series):
    return (series < 2).astype(int)

def ratio_digits_url_increment_condition(series):
    return series.isin(["High"])

def nb_subdomains_increment_condition(series):
    return series.isin(["High"])

def length_words_raw_increment_condition(series):
    return series.isin(["High", "Very High"])

def char_repeat_increment_condition(series):
    return series.isin(["High", "Very High"])

def shortest_word_host_increment_condition(series):
    return series.isin(["Very High"])

def nb_slash_increment_condition(series):
    return series.isin(["High", "Very High"])

def longest_word_host_increment_condition(series):
    return series.isin(["High", "Very High"])

def avg_word_host_increment_condition(series):
    return series.isin(["High", "Very High"])

# =============================================================================
#                         Clase AddWeirdColumn
# =============================================================================

class AddWeirdColumn(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_modify):
        self.features_to_modify = features_to_modify
        self.operations = self._define_operations()
        self.bin_params_ = {}
        self.generated_features_ = []
        self.features_faltantes_ = []  # Track de features no encontradas

    def _define_operations(self):
        """
        Define todas las reglas de transformación, evitando lambdas directas
        para poder serializar (pickle) correctamente.
        """
        return {
            # --- Transformaciones condicionales ---
            # Crea una columna binaria que indica si cumple la condicion o no. 
            # Si ademas es increment=True, suma 1 a 'is_weird'
            'nb_at': {'type': 'binary', 'threshold': 0, 'increment': True},
            'nb_eq': {'type': 'binary', 'threshold': 0, 'increment': False},
            'nb_and': {'type': 'binary', 'threshold': 0, 'increment': False},
            'nb_star': {'type': 'binary', 'threshold': 0, 'increment': False},
            'nb_colon': {'type': 'binary', 'threshold': 2, 'increment': False},
            'nb_semicolumn': {'type': 'binary', 'threshold': 1, 'increment': False},
            'nb_dollar': {'type': 'binary', 'threshold': 0, 'increment': False},
            'nb_com': {'type': 'binary', 'threshold': 0, 'increment': False},
            'nb_dslash': {'type': 'binary', 'threshold': 0, 'increment': False},
            'http_in_path': {'type': 'binary', 'threshold': 0, 'increment': False},
            'ratio_digits_host': {'type': 'binary', 'threshold': 0.10, 'increment': True},
            'tld_in_subdomain': {'type': 'binary', 'threshold': 0, 'increment': False},
            'abnormal_subdomain': {'type': 'binary', 'threshold': 0, 'increment': False},
            'prefix_suffix': {'type': 'binary', 'threshold': 1, 'increment': True},
            'nb_external_redirection': {'type': 'binary', 'threshold': 1, 'increment': False},
            'longest_words_raw': {'type': 'binary', 'threshold': 24, 'increment': False},
            'length_url': {'type': 'binary', 'threshold': 1098, 'increment': False},
            'length_hostname': {'type': 'binary', 'threshold': 74, 'increment': False},
            'ip': {'type': 'binary', 'threshold': 1, 'increment': False},
            'nb_dots': {'type': 'binary', 'threshold': 4, 'increment': True},
            'longest_word_path': {'type': 'binary', 'threshold': 16, 'increment': True},
            'avg_words_raw': {'type': 'binary', 'threshold': 13, 'increment': False},
            'avg_word_path': {'type': 'binary', 'threshold': 30, 'increment': False},
            'phish_hints': {'type': 'binary', 'threshold': 1, 'increment': True},
            'brand_in_subdomain': {'type': 'binary', 'threshold': 1, 'increment': False},
            'brand_in_path': {'type': 'binary', 'threshold': 1, 'increment': False},
            'suspecious_tld': {'type': 'binary', 'threshold': 1, 'increment': True},
            'statistical_report': {'type': 'binary', 'threshold': 1, 'increment': True},
            'empty_title': {'type': 'binary', 'threshold': 1, 'increment': False},
            'web_traffic': {'type': 'binary', 'threshold': 0, 'increment': False},
            'dns_record': {'type': 'binary', 'threshold': 1, 'increment': False},
            'google_index': {'type': 'binary', 'threshold': 1, 'increment': False},
            'nb_www': {'type': 'binary', 'threshold': 1, 'increment': True},

            # --- Variables categóricas ---
            'ratio_digits_url': {
                'type': 'categorical',
                'bins': 3,
                'labels': ["Low", "Medium", "High"],
                'increment_condition': ratio_digits_url_increment_condition
            },
            'nb_subdomains': {
                'type': 'categorical',
                'bins': 3,
                'labels': ["Low", "Medium", "High"],
                'increment_condition': nb_subdomains_increment_condition
            },
            'length_words_raw': {
                'type': 'categorical',
                'bins': 5,
                'labels': ["Very Low", "Low", "Medium", "High", "Very High"],
                'increment_condition': length_words_raw_increment_condition
            },
            'char_repeat': {
                'type': 'categorical',
                'bins': 5,
                'labels': ["Very Low", "Low", "Medium", "High", "Very High"],
                'increment_condition': char_repeat_increment_condition
            },
            'shortest_word_host': {
                'type': 'categorical',
                'bins': 5,
                'labels': ["Very Low", "Low", "Medium", "High", "Very High"],
                'increment_condition': shortest_word_host_increment_condition
            },
            'nb_slash': {
                'type': 'categorical',
                'bins': 5,
                'labels': ["Very Low", "Low", "Medium", "High", "Very High"],
                'increment_condition': nb_slash_increment_condition
            },
            'longest_word_host': {
                'type': 'categorical',
                'bins': 5,
                'labels': ["Very Low", "Low", "Medium", "High", "Very High"],
                'increment_condition': longest_word_host_increment_condition
            },
            'avg_word_host': {
                'type': 'categorical',
                'bins': 5,
                'labels': ["Very Low", "Low", "Medium", "High", "Very High"],
                'increment_condition': avg_word_host_increment_condition
            },

            # --- Casos especiales ---
            'page_rank': {
                'type': 'custom',
                'condition': page_rank_condition
            },

            # --- Variables sin transformación ---
            'https_token': {'type': 'passthrough'},
            'safe_anchor': {'type': 'passthrough'},
            'shortest_word_path': {'type': 'passthrough'},
            'domain_in_title': {'type': 'passthrough'},
            'domain_with_copyright': {'type': 'passthrough'}
        }

    def _categorize_column(self, X, feature, bins, labels):
        """Aplica discretización usando bins aprendidos."""
        if feature not in self.bin_params_:
            raise ValueError(f"Bins para {feature} no calculados en fit().")
        return pd.cut(
            X[feature],
            bins=self.bin_params_[feature],
            labels=labels[:len(self.bin_params_[feature]) - 1],  # Ajustar etiquetas al número real de bins
            include_lowest=True
        )

    def fit(self, X, y=None):
        """Calcula parámetros de discretización."""
        self.features_modified_ = []  # Resetear al inicio de fit
        for feature, op in self.operations.items():
            if op['type'] == 'categorical' and feature in X.columns:
                try:
                    _, bins = pd.qcut(
                        X[feature],
                        q=op['bins'],
                        retbins=True,
                        duplicates='drop'
                    )
                    self.bin_params_[feature] = bins
                except ValueError as e:
                    warnings.warn(f"Error en {feature}: {str(e)}", UserWarning)
        return self

    def transform(self, X):
        X = X.copy()  # Asegúrate de que no estás modificando el original
        X['is_weird'] = 0
        self.generated_features_ = []  # Resetear en cada transform

        for feature in self.features_to_modify:
            if feature not in self.operations:
                # Si la feature no está en operaciones, continuar
                continue

            op = self.operations[feature]
            self.features_modified_.append(feature)  # Registrar feature procesada

            # --- Aplicar transformación según tipo ---
            if op['type'] == 'binary':
                mask = (X[feature] > op['threshold']).astype(int)
                if op['increment']:
                    X['is_weird'] += mask
                X[f'{feature}_bin'] = mask
                self.generated_features_.append(f'{feature}_bin')  # Binaria

            elif op['type'] == 'categorical':
                bins = self.bin_params_.get(feature, None)
                labels = op['labels']
                if bins is None:
                    raise ValueError(f"No se calcularon bins para '{feature}'.")

                adjusted_labels = labels[:len(bins)-1]
                try:
                    X[f'{feature}_interval'] = pd.cut(
                        X[feature],
                        bins=bins,
                        labels=adjusted_labels,
                        include_lowest=True
                    )
                    print(f"Columna generada: {feature}_interval")

                    if 'increment_condition' in op:
                        # Suma 1 a is_weird si se cumple la condición 
                        X['is_weird'] += op['increment_condition'](
                            X[f'{feature}_interval']
                        ).astype(int)
                    self.generated_features_.append(f'{feature}_interval')  # Categórica
                except Exception as e:
                    warnings.warn(f"Error en {feature}: {str(e)}", UserWarning)

            elif op['type'] == 'custom':
                X['is_weird'] += op['condition'](X[feature]).astype(int)

            elif op['type'] == 'passthrough':
                # No se realiza transformación adicional ni se incrementa is_weird
                pass

        # --- Devuelve el DataFrame transformado ---
        return X


# --- Clase para codificar variables categóricas ---
class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Aplica One-Hot Encoding a features categóricas, manejando valores desconocidos.

    Args:
        categorical_features (list): Lista de columnas categóricas.
    """
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoder = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        if self.categorical_features:
            self.encoder.fit(X[self.categorical_features])
            self.feature_names_out_ = self.encoder.get_feature_names_out(self.categorical_features)
        return self

    def transform(self, X):
        if self.categorical_features:
            encoded = self.encoder.transform(X[self.categorical_features])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.feature_names_out_,
                index=X.index
            )
            return pd.concat([X.drop(columns=self.categorical_features), encoded_df], axis=1)
        return X


# --- Wrapper para XGBoost optimizado ---
class SklearnXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper de XGBClassifier para compatibilidad total con Scikit-learn, con soporte para eval_set.
    """
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.classes_ = None  # Necesario para sklearn
        self.fit_params = {}  # Parámetros adicionales para el método fit

    def set_eval_set(self, X_validation, y_validation):
        """
        Configura el conjunto de validación y otros parámetros específicos de XGBoost.
        """
        self.fit_params = {
            "eval_set": [(X_validation, y_validation)],
            "early_stopping_rounds": 10,  # Número de iteraciones para detener
            "verbose": False  # Silenciar la salida de logs
        }

    def fit(self, X, y, **kwargs):
        """Entrena el modelo y actualiza las clases."""
        fit_params = self.fit_params.copy()
        fit_params.update(kwargs)  # Permitir sobrescribir parámetros si es necesario
        self.model.fit(X, y, **fit_params)
        self.classes_ = np.unique(y)  # Actualiza las clases
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        """Propiedad para acceder a las importancias de características del modelo subyacente."""
        return self.model.feature_importances_

class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica StandardScaler y retorna un DataFrame con los nombres de columnas originales.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None  # Para almacenar los nombres de las columnas

    def fit(self, X, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        return self

    def transform(self, X):
        scaled_array = self.scaler.transform(X)
        return pd.DataFrame(scaled_array, columns=self.columns, index=X.index)

class RemoveBinaryDuplicates(BaseEstimator, TransformerMixin):
    """
    Elimina columnas redundantes en variables binarias/categóricas que son mutuamente excluyentes.
    Detecta pares de columnas que siempre suman 1 y elimina la redundante.
    """
    def fit(self, X, y=None):
        self.redundant_cols_ = []
        binary_cols = X.select_dtypes(include=['int', 'float']).columns

        for col in binary_cols:
            # Detectar columnas complementarias que suman 1
            complement_cols = [other_col for other_col in binary_cols if all((X[col] + X[other_col]) == 1)]
            if complement_cols:
                self.redundant_cols_.extend(complement_cols)  # Guardar las columnas redundantes

        self.redundant_cols_ = list(set(self.redundant_cols_))  # Eliminar duplicados en la lista
        return self

    def transform(self, X):
        # Verificar las columnas que efectivamente serán eliminadas
        cols_to_remove = [col for col in self.redundant_cols_ if col in X.columns]
        if cols_to_remove:
            print(f"Columnas redundantes eliminadas: {cols_to_remove}")
        return X.drop(columns=cols_to_remove, errors='ignore')


