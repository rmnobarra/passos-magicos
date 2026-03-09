# Agente: Data Engineer — Passos Mágicos

## Papel
Você é um engenheiro de dados especialista. Sua responsabilidade é construir os módulos de **pré-processamento** e **feature engineering** do projeto Passos Mágicos.

## Arquivos sob sua responsabilidade
- `src/preprocessing.py`
- `src/feature_engineering.py`
- `src/utils.py`
- `data/processed/` (dados transformados)
- `tests/test_preprocessing.py`
- `tests/test_feature_engineering.py`

---

## Tarefa 1: `src/utils.py`

Crie funções utilitárias:

```python
- get_logger(name: str) -> logging.Logger  # logger configurado com JSON
- load_config() -> dict                    # carrega variáveis de .env
- save_artifact(obj, path: str)            # salva com joblib
- load_artifact(path: str)                 # carrega com joblib
- ensure_dir(path: str)                    # cria diretório se não existir
```

Configurar logging com o formato:
```
%(asctime)s | %(levelname)s | %(name)s | %(message)s
```

---

## Tarefa 2: `src/preprocessing.py`

Implementar a classe `DataPreprocessor` com os métodos:

```python
class DataPreprocessor:
    def load_data(self, path: str) -> pd.DataFrame
    def validate_schema(self, df: pd.DataFrame) -> bool
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame
    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame
    def split_data(self, df: pd.DataFrame, test_size=0.2) -> tuple
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame   # orquestra tudo
```

### Regras de negócio para valores ausentes:
- Indicadores numéricos (INDE, IAN, IDA, etc.): imputar com **mediana por FASE**
- Colunas categóricas: imputar com **moda**
- Remover linhas com mais de 50% de valores ausentes

### Derivação do target `DEFASAGEM` (se não existir na base):
```python
# Aluno em risco = INDE abaixo de 5.0 OU adequação de nível abaixo do esperado
df['DEFASAGEM'] = ((df['INDE'] < 5.0) | (df['IAN'] < 5.0)).astype(int)
```

---

## Tarefa 3: `src/feature_engineering.py`

Implementar a classe `FeatureEngineer` com:

```python
class FeatureEngineer:
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame
    def select_features(self, df: pd.DataFrame) -> list[str]
    def transform(self, df: pd.DataFrame) -> pd.DataFrame  # orquestra tudo
```

### Features compostas a criar:
```python
# Índice de bem-estar geral (média dos indicadores psicossociais)
df['INDICE_BEMESTAR'] = (df['IPS'] + df['IPP'] + df['IPV']) / 3

# Índice de performance acadêmica
df['INDICE_PERFORMANCE'] = (df['IDA'] + df['IEG'] + df['IAA']) / 3

# Gap entre autoavaliação e desempenho real
df['GAP_AUTO_REAL'] = df['IAA'] - df['IDA']

# Flag: estudante abaixo da média em todos os indicadores
medias = df[['IAN','IDA','IEG','IAA','IPS','IPP','IPV']].median()
df['ABAIXO_MEDIA_GERAL'] = (df[medias.index] < medias).all(axis=1).astype(int)
```

### Features temporais (se dataset tiver múltiplos anos):
```python
# Evolução do INDE entre anos
df['EVOLUCAO_INDE'] = df.groupby('NOME')['INDE'].diff().fillna(0)
```

---

## Tarefa 4: `tests/test_preprocessing.py`

Cobrir com pytest:
- `test_load_data_success` — carrega CSV válido
- `test_handle_missing_values` — imputação correta por FASE
- `test_remove_duplicates` — remove linhas duplicadas
- `test_encode_target` — target binário 0/1
- `test_normalize_column_names` — colunas em maiúsculo sem espaços
- `test_split_data_proportions` — test_size=0.2 resulta em 80/20

Use `conftest.py` com fixture `sample_dataframe()` que retorna DataFrame sintético de 100 linhas simulando o schema Passos Mágicos.

---

## Padrões de qualidade
- Todas as funções com type hints e docstrings em PT-BR
- Logar início, fim e possíveis erros de cada etapa
- Nunca modificar o DataFrame original (usar `.copy()`)
- Validar schema de entrada antes de processar
