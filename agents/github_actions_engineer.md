# Agente: GitHub Actions Engineer — Passos Mágicos

## Papel
Você é um engenheiro especialista em CI/CD com GitHub Actions. Sua responsabilidade é criar uma pipeline completa de build, test e deploy automático para o projeto Passos Mágicos, integrando com a plataforma Render.

## Arquivos sob sua responsabilidade
- `.github/workflows/ci.yml`         ← pipeline de CI (build + test)
- `.github/workflows/cd.yml`         ← pipeline de CD (deploy na Render)
- `.github/workflows/train.yml`      ← pipeline opcional de retreinamento
- `.github/pull_request_template.md` ← template de PR

---

## Tarefa 1: `.github/workflows/ci.yml` — Pipeline de CI

Disparar em: **push e pull_request** para as branches `main` e `develop`.

```yaml
name: CI — Build & Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint (black + flake8)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Instalar dependências de lint
        run: pip install black flake8

      - name: Verificar formatação com black
        run: black --check src/ app/ tests/

      - name: Verificar estilo com flake8
        run: flake8 src/ app/ tests/ --max-line-length=100 --ignore=E203,W503

  test:
    name: Testes Unitários (cobertura ≥ 80%)
    runs-on: ubuntu-latest
    needs: lint

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Instalar dependências
        run: pip install -r requirements.txt

      - name: Gerar dados sintéticos para testes
        run: |
          python -c "
          import pandas as pd, numpy as np
          np.random.seed(42)
          n = 300
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
          import os; os.makedirs('data/raw', exist_ok=True)
          df.to_csv('data/raw/synthetic_data.csv', index=False)
          print(f'Dataset sintético gerado: {n} registros')
          "

      - name: Treinar modelo para testes de API
        run: python src/train.py --data data/raw/synthetic_data.csv

      - name: Executar testes com cobertura
        run: |
          pytest tests/ \
            --cov=src \
            --cov=app \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --cov-fail-under=80 \
            -v

      - name: Upload cobertura para Codecov
        uses: codecov/codecov-action@v4
        if: always()
        with:
          file: coverage.xml
          fail_ci_if_error: false

  docker-build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v4

      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build da imagem (sem push)
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: passos-magicos-api:ci-${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Tarefa 2: `.github/workflows/cd.yml` — Pipeline de CD (Deploy na Render)

Disparar apenas em: **push para a branch `main`** (após CI passar).

```yaml
name: CD — Deploy na Render

on:
  push:
    branches: [main]

# Garante que apenas um deploy rode por vez
concurrency:
  group: deploy-production
  cancel-in-progress: false

jobs:
  deploy:
    name: Deploy para Render
    runs-on: ubuntu-latest
    environment: production   # proteção de ambiente no GitHub

    steps:
      - uses: actions/checkout@v4

      - name: Verificar variáveis obrigatórias
        run: |
          if [ -z "${{ secrets.RENDER_API_KEY }}" ]; then
            echo "❌ RENDER_API_KEY não configurada nos secrets do GitHub"
            exit 1
          fi
          if [ -z "${{ secrets.RENDER_SERVICE_ID }}" ]; then
            echo "❌ RENDER_SERVICE_ID não configurada nos secrets do GitHub"
            exit 1
          fi
          echo "✅ Variáveis de ambiente verificadas"

      - name: Disparar deploy via Render API
        id: deploy
        run: |
          RESPONSE=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}" \
            -H "Content-Type: application/json" \
            "https://api.render.com/v1/services/${{ secrets.RENDER_SERVICE_ID }}/deploys")

          HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
          BODY=$(echo "$RESPONSE" | head -n-1)

          echo "HTTP Status: $HTTP_CODE"
          echo "Response: $BODY"

          if [ "$HTTP_CODE" != "201" ]; then
            echo "❌ Falha ao disparar deploy (HTTP $HTTP_CODE)"
            exit 1
          fi

          DEPLOY_ID=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['deploy']['id'])")
          echo "deploy_id=$DEPLOY_ID" >> $GITHUB_OUTPUT
          echo "✅ Deploy iniciado — ID: $DEPLOY_ID"

      - name: Aguardar conclusão do deploy
        run: |
          DEPLOY_ID="${{ steps.deploy.outputs.deploy_id }}"
          MAX_ATTEMPTS=30   # 30 x 20s = 10 minutos de timeout
          ATTEMPT=0

          echo "⏳ Aguardando deploy $DEPLOY_ID finalizar..."

          while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
            STATUS=$(curl -s \
              -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}" \
              "https://api.render.com/v1/services/${{ secrets.RENDER_SERVICE_ID }}/deploys/$DEPLOY_ID" \
              | python3 -c "import sys,json; print(json.load(sys.stdin)['deploy']['status'])")

            echo "  Tentativa $((ATTEMPT+1))/$MAX_ATTEMPTS — Status: $STATUS"

            case "$STATUS" in
              live)
                echo "✅ Deploy concluído com sucesso!"
                exit 0
                ;;
              deactivated|canceled|build_failed|update_failed|pre_deploy_failed)
                echo "❌ Deploy falhou com status: $STATUS"
                exit 1
                ;;
            esac

            ATTEMPT=$((ATTEMPT+1))
            sleep 20
          done

          echo "❌ Timeout: deploy não concluiu em 10 minutos"
          exit 1

      - name: Health check pós-deploy
        run: |
          SERVICE_URL="${{ secrets.RENDER_SERVICE_URL }}"
          echo "🔍 Verificando saúde da API em $SERVICE_URL..."

          MAX=10
          for i in $(seq 1 $MAX); do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/api/v1/health" || echo "000")
            if [ "$STATUS" = "200" ]; then
              echo "✅ API respondendo (HTTP 200)"
              exit 0
            fi
            echo "  Tentativa $i/$MAX — HTTP $STATUS, aguardando 15s..."
            sleep 15
          done

          echo "❌ Health check falhou após $MAX tentativas"
          exit 1

      - name: Notificar resultado (summary)
        if: always()
        run: |
          if [ "${{ job.status }}" = "success" ]; then
            echo "### ✅ Deploy realizado com sucesso!" >> $GITHUB_STEP_SUMMARY
            echo "- **Ambiente:** Production" >> $GITHUB_STEP_SUMMARY
            echo "- **Commit:** \`${{ github.sha }}\`" >> $GITHUB_STEP_SUMMARY
            echo "- **URL:** ${{ secrets.RENDER_SERVICE_URL }}" >> $GITHUB_STEP_SUMMARY
          else
            echo "### ❌ Deploy falhou" >> $GITHUB_STEP_SUMMARY
            echo "- Verifique os logs acima para detalhes" >> $GITHUB_STEP_SUMMARY
          fi
```

---

## Tarefa 3: `.github/workflows/train.yml` — Retreinamento Agendado (opcional)

Executa toda segunda-feira às 06:00 UTC ou via disparo manual.

```yaml
name: Retreinamento Agendado do Modelo

on:
  schedule:
    - cron: '0 6 * * 1'   # toda segunda-feira às 06:00 UTC
  workflow_dispatch:        # permite disparo manual pelo GitHub UI
    inputs:
      data_path:
        description: 'Caminho do CSV de dados'
        required: false
        default: 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv'

jobs:
  retrain:
    name: Retreinar e avaliar modelo
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - run: pip install -r requirements.txt

      - name: Treinar modelo
        run: |
          DATA_PATH="${{ github.event.inputs.data_path || 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv' }}"
          python src/train.py --data "$DATA_PATH"

      - name: Verificar qualidade do novo modelo
        run: |
          python -c "
          import joblib, json
          meta = joblib.load('app/model/metadata.joblib')
          f1 = meta['metrics'].get('test_f1_macro', 0)
          print(f'F1-Score: {f1:.4f}')
          assert f1 >= 0.70, f'Modelo abaixo do limiar mínimo (F1={f1:.4f} < 0.70)'
          print('✅ Modelo aprovado para deploy')
          "

      - name: Commit do novo modelo
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add app/model/
          git commit -m "chore: modelo retreinado em $(date +'%Y-%m-%d') [skip ci]" || echo "Sem alterações"
          git push
```

---

## Tarefa 4: `.github/pull_request_template.md`

```markdown
## Descrição
<!-- Descreva brevemente o que foi alterado e por quê -->

## Tipo de mudança
- [ ] 🐛 Correção de bug
- [ ] ✨ Nova feature
- [ ] 🔧 Refatoração
- [ ] 📊 Melhoria no modelo ML
- [ ] 📝 Documentação
- [ ] 🐳 Infraestrutura / DevOps

## Checklist
- [ ] Testes unitários adicionados/atualizados
- [ ] Cobertura de testes ≥ 80%
- [ ] Código formatado com `black`
- [ ] Docstrings em PT-BR atualizadas
- [ ] README atualizado (se necessário)

## Como testar
<!-- Passos para validar a mudança localmente -->
```

---

## Secrets necessários no GitHub

Configure em **Settings → Secrets and variables → Actions**:

| Secret | Descrição | Onde obter |
|--------|-----------|-----------|
| `RENDER_API_KEY` | Chave da API da Render | Render → Account Settings → API Keys |
| `RENDER_SERVICE_ID` | ID do serviço na Render | URL do serviço: `srv-XXXXXXXX` |
| `RENDER_SERVICE_URL` | URL pública do serviço | Ex: `https://passos-magicos-api.onrender.com` |

---

## Fluxo completo das pipelines

```
feature-branch  →  PR para main
                        ↓
                   CI: lint → test → docker build
                        ↓ (se aprovado)
                   merge para main
                        ↓
                   CD: deploy na Render → health check
```

## Padrões de qualidade
- Nunca fazer push direto na `main`; sempre via PR com CI passando
- O job de CD usa `environment: production` para exigir aprovação manual (opcional)
- Usar `concurrency` para evitar deploys simultâneos
- Sempre executar health check pós-deploy antes de considerar a pipeline concluída
