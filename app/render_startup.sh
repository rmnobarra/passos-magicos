#!/bin/bash
set -e

echo "🚀 Iniciando Passos Mágicos API..."
echo "   Ambiente: ${APP_ENV:-production}"
echo "   Python: $(python --version)"

MODEL_PATH="${MODEL_PATH:-/app/app/model/model.joblib}"
METADATA_PATH="${METADATA_PATH:-/app/app/model/metadata.joblib}"

# Verificar se o modelo já existe no disco persistente
if [ ! -f "$MODEL_PATH" ]; then
  echo "⚠️  Modelo não encontrado em $MODEL_PATH"
  echo "🔧 Gerando dados sintéticos e treinando modelo inicial..."

  python -c "
import pandas as pd, numpy as np, os
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'NOME': [f'EST_{i:03d}' for i in range(n)],
    'ANO': np.random.choice([2022,2023,2024], n),
    'FASE': np.random.randint(0,9,n),
    'TURMA': np.random.choice(['A','B','C'], n),
    'INDE': np.random.uniform(2,10,n),
    'IAN': np.random.uniform(2,10,n),
    'IDA': np.random.uniform(2,10,n),
    'IEG': np.random.uniform(2,10,n),
    'IAA': np.random.uniform(2,10,n),
    'IPS': np.random.uniform(2,10,n),
    'IPP': np.random.uniform(2,10,n),
    'IPV': np.random.uniform(2,10,n),
    'PONTO_DE_VIRADA': np.random.choice([0,1], n),
})
df['DEFASAGEM'] = ((df['INDE'] < 5.0) | (df['IAN'] < 5.0)).astype(int)
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/synthetic_data.csv', index=False)
print(f'Dataset sintético: {n} registros')
"

  python src/train.py --data data/raw/synthetic_data.csv
  echo "✅ Modelo treinado e salvo em $MODEL_PATH"
else
  echo "✅ Modelo encontrado: $MODEL_PATH"
  # Exibir versão e métricas do modelo existente para rastreabilidade
  python -c "
import joblib
try:
    meta = joblib.load('$METADATA_PATH')
    print(f'   Versão  : {meta.get(\"version\", \"N/A\")}')
    print(f'   Algoritmo: {meta.get(\"algorithm\", \"N/A\")}')
    f1 = meta.get('metrics', {}).get('test_f1_macro', 'N/A')
    print(f'   F1 macro : {f1}')
except Exception as e:
    print(f'   (metadata indisponível: {e})')
" 2>/dev/null || true
fi

# Criar diretório de logs
mkdir -p /app/logs

# Subir a API
echo "🌐 Subindo FastAPI na porta 8000..."
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info \
  --access-log
