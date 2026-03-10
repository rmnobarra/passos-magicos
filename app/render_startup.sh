#!/bin/bash
set -e

echo "🚀 Iniciando Passos Mágicos API..."
echo "   Ambiente: ${APP_ENV:-production}"
echo "   Python: $(python --version)"

MODEL_PATH="${MODEL_PATH:-/app/app/model/model.joblib}"
METADATA_PATH="${METADATA_PATH:-/app/app/model/metadata.joblib}"

# Verificar modelo (deve estar sempre presente — baked na imagem via git)
if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ Modelo não encontrado em $MODEL_PATH"
  echo "   Certifique-se de que app/model/model.joblib está commitado no repositório."
  exit 1
fi

# Exibir versão e métricas do modelo para rastreabilidade nos logs
python -c "
import joblib
try:
    meta = joblib.load('$METADATA_PATH')
    print(f'   Versão   : {meta.get(\"version\", \"N/A\")}')
    print(f'   Algoritmo: {meta.get(\"algorithm\", \"N/A\")}')
    f1 = meta.get('metrics', {}).get('test_f1_macro', 'N/A')
    trained = meta.get('trained_at', 'N/A')
    print(f'   F1 macro : {f1}')
    print(f'   Treinado : {trained}')
except Exception as e:
    print(f'   (metadata indisponível: {e})')
" 2>/dev/null || true

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
