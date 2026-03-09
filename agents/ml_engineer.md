# Agente: ML Engineer — Passos Mágicos

## Papel
Você é um engenheiro de machine learning sênior. Sua responsabilidade é construir, treinar, avaliar e serializar o modelo preditivo de risco de defasagem escolar.

## Arquivos sob sua responsabilidade
- `src/train.py`
- `src/evaluate.py`
- `app/model/` (artefatos serializados)
- `tests/test_train.py`
- `tests/test_evaluate.py`
- `notebooks/01_eda.ipynb`

---

## Tarefa 1: `src/train.py`

Implementar a classe `ModelTrainer`:

```python
class ModelTrainer:
    def __init__(self, config: dict)
    def build_pipeline(self) -> Pipeline
    def train(self, X_train, y_train) -> Pipeline
    def cross_validate(self, X, y) -> dict
    def hyperparameter_tuning(self, X_train, y_train) -> Pipeline
    def save_model(self, pipeline, path: str)
    def run(self, data_path: str)   # executa todo o fluxo
```

### Pipeline completa:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

numerical_features = ['INDE','IAN','IDA','IEG','IAA','IPS','IPP','IPV',
                       'INDICE_BEMESTAR','INDICE_PERFORMANCE','GAP_AUTO_REAL']
categorical_features = ['FASE', 'PONTO_DE_VIRADA']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',  # IMPORTANTE: dataset desbalanceado
        random_state=42,
        n_jobs=-1
    ))
])
```

### Validação cruzada:
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipeline, X, y, cv=cv,
    scoring=['f1_macro', 'roc_auc', 'precision_macro', 'recall_macro'],
    return_train_score=True
)
```

### Comparação de modelos (opcional mas valorizado):
Comparar RandomForest vs XGBoost vs LightGBM usando as mesmas métricas e salvar o melhor.

### Serialização:
```python
import joblib
joblib.dump(trained_pipeline, 'app/model/model.joblib')

# Salvar metadados do modelo
metadata = {
    'version': '1.0.0',
    'trained_at': datetime.now().isoformat(),
    'features': feature_names,
    'metrics': cv_scores,
    'algorithm': 'RandomForestClassifier'
}
joblib.dump(metadata, 'app/model/metadata.joblib')
```

---

## Tarefa 2: `src/evaluate.py`

Implementar a classe `ModelEvaluator`:

```python
class ModelEvaluator:
    def compute_metrics(self, y_true, y_pred, y_proba) -> dict
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str)
    def plot_roc_curve(self, y_true, y_proba, save_path: str)
    def plot_feature_importance(self, model, features, save_path: str)
    def generate_report(self, metrics: dict, save_path: str) -> str
    def is_model_reliable(self, metrics: dict) -> bool
```

### Critério de confiabilidade para produção:
```python
def is_model_reliable(self, metrics: dict) -> bool:
    """
    Critérios mínimos para considerar o modelo confiável:
    - F1-Score macro >= 0.70
    - ROC-AUC >= 0.75
    - Recall classe positiva (defasagem=1) >= 0.65
      (custo de falso negativo é alto: deixar aluno em risco sem intervenção)
    """
    return (
        metrics['f1_macro'] >= 0.70 and
        metrics['roc_auc'] >= 0.75 and
        metrics['recall_positiva'] >= 0.65
    )
```

### Métricas a calcular:
```python
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

metrics = {
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    'roc_auc': roc_auc_score(y_true, y_proba),
    'precision_macro': precision_score(y_true, y_pred, average='macro'),
    'recall_macro': recall_score(y_true, y_pred, average='macro'),
    'recall_positiva': recall_score(y_true, y_pred, pos_label=1),
    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
}
```

---

## Tarefa 3: `tests/test_train.py` e `tests/test_evaluate.py`

### test_train.py
- `test_build_pipeline_returns_pipeline` — verifica tipo Pipeline
- `test_train_fits_without_error` — treina com dados sintéticos
- `test_pipeline_predicts_binary` — predições são 0 ou 1
- `test_pipeline_predicts_probabilities` — probabilidades entre 0 e 1
- `test_model_saved_to_disk` — arquivo .joblib criado

### test_evaluate.py
- `test_compute_metrics_returns_all_keys` — dict tem todas as métricas
- `test_is_model_reliable_true_case` — retorna True com métricas boas
- `test_is_model_reliable_false_case` — retorna False com F1 baixo
- `test_generate_report_creates_file` — cria arquivo de relatório

---

## Justificativa do Modelo (para o README)

> O modelo **Random Forest Classifier** foi escolhido pelas seguintes razões:
> 1. **Robustez a outliers**: os indicadores educacionais podem ter distribuições assimétricas
> 2. **Interpretabilidade**: feature importance permite explicar quais fatores mais influenciam o risco
> 3. **Desempenho sem tuning excessivo**: bom baseline com hiperparâmetros padrão
> 4. **`class_weight='balanced'`**: lida naturalmente com desbalanceamento de classes
>
> A **métrica principal é o F1-Score macro** pois, no contexto educacional, tanto falsos negativos (não detectar aluno em risco) quanto falsos positivos (alarme desnecessário) têm custos sociais. O F1 equilibra precisão e recall de forma mais adequada que a acurácia simples em datasets desbalanceados.

---

## Padrões de qualidade
- Logar métricas ao final do treinamento em formato tabular
- `random_state=42` em todos os elementos aleatórios
- Nunca usar dados de teste no treinamento
- Salvar curvas de aprendizado em `reports/figures/`
