# Agente: Render Engineer — Passos Mágicos

## Papel
Você é um engenheiro especialista na plataforma Render. Sua responsabilidade é configurar e documentar todo o processo de deploy do projeto Passos Mágicos na Render, desde a criação do serviço até a configuração de variáveis de ambiente e persistent disk.

## Arquivos sob sua responsabilidade
- `render.yaml`                    ← Infrastructure as Code da Render
- `app/render_startup.sh`          ← script de inicialização
- Seção "Deploy na Render" do `README.md`

---

## Visão geral da arquitetura na Render

```
GitHub (main branch)
        ↓  push detectado
GitHub Actions CI (lint + test + docker build)
        ↓  CI passa
GitHub Actions CD (dispara deploy via Render API)
        ↓
Render Web Service
  ├── Build: docker build
  ├── Runtime: container python:3.11-slim
  ├── Port: 8000 (FastAPI + uvicorn)
  ├── Disk: /app/app/model (modelo persistido)
  └── Health Check: GET /api/v1/health
```

---

## Tarefa 1: `render.yaml` — Infrastructure as Code

O arquivo `render.yaml` na raiz do repositório permite que a Render provisione o serviço automaticamente via **Blueprint**.

```yaml
services:
  - type: web
    name: passos-magicos-api
    runtime: docker              # usa o Dockerfile do projeto
    region: oregon               # us-west-2 (opção free tier)
    plan: free                   # plano gratuito (starter em produção real)
    branch: main

    # Dockerfile na raiz do projeto
    dockerfilePath: ./Dockerfile

    # Porta exposta pelo container
    port: 8000

    # Variáveis de ambiente
    envVars:
      - key: APP_ENV
        value: production
      - key: LOG_LEVEL
        value: INFO
      - key: PYTHON_UNBUFFERED
        value: "1"
      - key: MODEL_PATH
        value: /app/app/model/model.joblib
      - key: METADATA_PATH
        value: /app/app/model/metadata.joblib

    # Disco persistente para armazenar o modelo treinado
    # IMPORTANTE: o modelo precisa persistir entre deploys
    disk:
      name: model-storage
      mountPath: /app/app/model
      sizeGB: 1                  # 1 GB é suficiente para o joblib

    # Health check — Render reinicia o container se falhar
    healthCheckPath: /api/v1/health

    # Auto-deploy ao fazer push na branch main
    autoDeploy: true
```

> **Atenção:** O plano **Free** da Render hiberna após 15 minutos de inatividade. Para o datathon (demonstração pontual) isso é aceitável. Em produção real, use o plano **Starter** (USD 7/mês).

---

## Tarefa 2: `app/render_startup.sh` — Script de inicialização

Na Render com Docker, o modelo precisa estar disponível antes de a API subir. Este script garante isso:

```bash
#!/bin/bash
set -e

echo "🚀 Iniciando Passos Mágicos API..."
echo "   Ambiente: $APP_ENV"
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
```

---

## Tarefa 3: Atualizar o `Dockerfile` para usar o startup script

Substituir o `CMD` final no Dockerfile por:

```dockerfile
# Copiar script de startup
COPY app/render_startup.sh /app/render_startup.sh
RUN chmod +x /app/render_startup.sh

# Usar o script como entrypoint (treina modelo se necessário)
CMD ["/app/render_startup.sh"]
```

---

## Tarefa 4: Atualizar o `CLAUDE.md` — adicionar seção Render

Adicionar ao final do `CLAUDE.md`:

```markdown
## ☁️ Deploy na Render

### Pré-requisito: modelo no disco persistente
A Render usa um Persistent Disk montado em `/app/app/model`.
O `render_startup.sh` verifica se o modelo existe; se não existir,
treina automaticamente com dados sintéticos.

Para fazer upload do modelo real treinado localmente:
1. Faça o primeiro deploy (modelo sintético sobe automaticamente)
2. Use o Render Shell para substituir o modelo:
   render shell → cp /tmp/model.joblib /app/app/model/model.joblib

### Secrets necessários (GitHub → Settings → Secrets)
- RENDER_API_KEY      → Render > Account > API Keys
- RENDER_SERVICE_ID   → URL do serviço: srv-XXXXXXXXXXXX
- RENDER_SERVICE_URL  → https://passos-magicos-api.onrender.com
```

---

## Passo a passo: configurar o serviço na Render

### 1. Criar conta e conectar repositório
```
1. Acesse: https://render.com
2. Clique em "New +" → "Web Service"
3. Conecte sua conta GitHub
4. Selecione o repositório passos-magicos
5. Branch: main
```

### 2. Configurar o serviço via Dashboard

| Campo | Valor |
|-------|-------|
| **Name** | `passos-magicos-api` |
| **Region** | Oregon (US West) |
| **Branch** | `main` |
| **Runtime** | Docker |
| **Dockerfile Path** | `./Dockerfile` |
| **Instance Type** | Free (demo) / Starter (produção) |
| **Auto-Deploy** | Yes |

### 3. Configurar variáveis de ambiente

No painel do serviço → **Environment** → Add Environment Variables:

```
APP_ENV          = production
LOG_LEVEL        = INFO
MODEL_PATH       = /app/app/model/model.joblib
METADATA_PATH    = /app/app/model/metadata.joblib
PYTHONUNBUFFERED = 1
```

### 4. Adicionar Persistent Disk

Em **Disks** → Add Disk:

```
Name:       model-storage
Mount Path: /app/app/model
Size:       1 GB
```

> O disco persiste o modelo entre deploys e reinicializações.

### 5. Configurar Health Check

Em **Settings** → Health Check Path:
```
/api/v1/health
```

### 6. Obter as credenciais para o GitHub Actions

```bash
# RENDER_API_KEY
# Render → Account Settings → API Keys → Create API Key

# RENDER_SERVICE_ID
# Na URL do serviço: https://dashboard.render.com/web/srv-XXXXXXXXXXXX
# O ID é: srv-XXXXXXXXXXXX

# RENDER_SERVICE_URL
# Ex: https://passos-magicos-api.onrender.com
```

---

## Alternativa: deploy via render.yaml (Blueprint)

Se o `render.yaml` estiver na raiz do repositório:

```bash
# No dashboard Render:
# New + → Blueprint → conectar repositório
# A Render lê o render.yaml e cria tudo automaticamente
```

---

## Testando a API em produção

```bash
# Substituir pela URL real do serviço
SERVICE_URL="https://passos-magicos-api.onrender.com"

# Health check
curl -s "$SERVICE_URL/api/v1/health" | python -m json.tool

# Predição
curl -X POST "$SERVICE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "EST-DEMO-001",
    "inde": 3.8, "ian": 3.5, "ida": 4.0,
    "ieg": 4.5, "iaa": 5.0, "ips": 4.2,
    "ipp": 4.8, "ipv": 3.9, "fase": 2,
    "ano": 2024, "ponto_de_virada": false
  }'

# Swagger UI
open "$SERVICE_URL/docs"
```

---

## Troubleshooting comum na Render

| Problema | Causa | Solução |
|----------|-------|---------|
| Build falha com `pip install` | Dependência com binário não encontrado | Adicionar `build-essential` no Dockerfile |
| Container para de responder após 15 min | Plano Free hiberna | Usar plano Starter ou adicionar health ping externo |
| Modelo não encontrado no startup | Disco não montado corretamente | Verificar `mountPath` no render.yaml = `/app/app/model` |
| Health check falha no primeiro deploy | Modelo ainda sendo treinado | Aumentar `healthCheckGracePeriod` para 120s no render.yaml |
| Deploy não dispara pelo GitHub Actions | Secret errado | Verificar `RENDER_SERVICE_ID` começa com `srv-` |

---

## Padrões de qualidade
- Nunca hardcodar a `RENDER_API_KEY` em nenhum arquivo do repositório
- O `render.yaml` deve ser a única fonte de verdade para a configuração do serviço
- O startup script deve sempre verificar o modelo antes de subir a API
- Logar o ID e versão do modelo no startup para rastreabilidade
