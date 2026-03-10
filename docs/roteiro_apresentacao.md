# Roteiro de Apresentação — Passos Mágicos
## Vídeo de 5 minutos · Banca Avaliadora

> **Duração total:** 5:00 · **Ritmo:** ~125 palavras/minuto · **Tom:** direto, confiante, com calor humano

---

## Distribuição de Tempo

| Bloco | Tema | Tempo |
|-------|------|-------|
| 1 | Abertura e problema | 0:45 |
| 2 | Solução em alto nível | 1:00 |
| 3 | Demo técnica ao vivo | 1:45 |
| 4 | Métricas e resultados | 0:45 |
| 5 | Encerramento e impacto | 0:45 |
| **Total** | | **5:00** |

---

## Bloco 1 — Abertura e contexto
### ⏱ 0:00–0:45 · 45 segundos

**Objetivo:** Conectar emocionalmente com o problema antes de falar de tecnologia.

**Fala:**
> "A Associação Passos Mágicos já transformou a vida de milhares de crianças
> em Embu-Guaçu. Mas com centenas de estudantes atendidos todo ano, uma
> pergunta crítica sempre fica sem resposta a tempo: qual aluno está prestes
> a cair em defasagem escolar?
>
> Identificar esse risco manualmente é lento, subjetivo e muitas vezes tarde
> demais. É exatamente esse problema que nosso projeto resolve."

**Na tela:**
- Logo da Associação Passos Mágicos
- Foto de estudantes em atividade (se disponível)
- Frase de impacto: _"32 anos transformando vidas. Como garantir que nenhuma seja perdida para a defasagem?"_

---

## Bloco 2 — A solução em alto nível
### ⏱ 0:45–1:45 · 60 segundos

**Objetivo:** Explicar o que foi construído sem entrar em código ainda.

**Fala:**
> "Construímos um sistema de Machine Learning que analisa os indicadores
> educacionais e psicossociais de cada estudante — como o INDE, engajamento,
> desempenho acadêmico e bem-estar — e calcula automaticamente o risco de
> defasagem.
>
> O resultado é entregue em tempo real via API: dado um estudante, o sistema
> responde em milissegundos se ele está em risco baixo, médio ou alto —
> e com qual probabilidade.
>
> Mas não paramos no modelo. Construímos todo o ciclo de vida de MLOps:
> pipeline de dados, treinamento automático, testes, deploy em nuvem e
> monitoramento contínuo de performance."

**Na tela:**
- Diagrama de fluxo simplificado:
  ```
  Dados (CSV) → Pipeline → Modelo → API REST → Decisão de intervenção
  ```
- Destaque: **18 features** analisadas por estudante
- Destaque: resposta em **< 100ms** via API

---

## Bloco 3 — Demonstração técnica ao vivo
### ⏱ 1:45–3:30 · 105 segundos

**Objetivo:** Mostrar o projeto funcionando. Seja concreto — mostre telas reais.

**Fala:**
> "Vamos ver funcionando.
>
> [mostrar estrutura de pastas]
> O projeto está organizado em módulos com responsabilidade única:
> pré-processamento, engenharia de features, treinamento, avaliação e API.
> Cada módulo é testado de forma independente.
>
> [mostrar terminal com pytest]
> Nossa suíte de testes garante mais de 80% de cobertura — padrão exigido
> para qualquer sistema que vai para produção.
>
> [mostrar Swagger UI com /predict]
> Aqui está a API rodando na Render. Vou fazer uma predição ao vivo:
> um estudante com INDE 3.8 e indicadores psicossociais baixos...
> o modelo retorna risco ALTO com 84% de probabilidade — em menos de
> 100 milissegundos.
>
> [mostrar GitHub Actions]
> A cada push no repositório, nossa pipeline de CI executa lint, testes e
> build da imagem Docker automaticamente. Se tudo passa, o deploy na Render
> acontece sem intervenção humana.
>
> [mostrar dashboard de monitoramento]
> E aqui o painel de monitoramento — acompanhamos drift nos dados e volume
> de predições em tempo real."

**Na tela (nesta ordem):**
1. Estrutura de diretórios (`src/`, `app/`, `tests/`, `monitoring/`)
2. Output do `pytest --cov` com cobertura ≥ 80%
3. Swagger UI → campo `/api/v1/predict` → executar predição ao vivo
4. GitHub Actions → pipeline verde (CI → CD)
5. Dashboard Streamlit com gráficos de drift

---

## Bloco 4 — Resultados e métricas
### ⏱ 3:30–4:15 · 45 segundos

**Objetivo:** Dar credibilidade ao modelo com números reais.

**Fala:**
> "Sobre a performance do modelo:
>
> Usamos LightGBM com validação cruzada estratificada em 5 folds.
> A métrica principal é o F1-Score macro — porque nesse contexto, um falso
> negativo tem custo social alto: deixar um aluno em risco sem intervenção.
>
> Nosso modelo atingiu F1-Score de **0.984** e ROC-AUC de **0.999** nos dados
> de teste — bem acima do limiar mínimo de 0.70 que definimos para considerar
> o modelo confiável para produção.
>
> Na matriz de confusão, de 106 estudantes em risco real, o modelo identificou
> 103 corretamente — apenas 3 falsos negativos em todo o conjunto de teste.
>
> As features mais relevantes foram INDE, IAN e o índice composto de
> bem-estar psicossocial — o que faz sentido com a missão da Passos Mágicos."

**Na tela:**
- Tabela de métricas reais do modelo:

  | Métrica | CV (5-fold) | Teste |
  |---------|-------------|-------|
  | F1-Score macro | 0.994 ± 0.002 | **0.984** |
  | ROC-AUC | 1.000 ± 0.000 | **0.999** |
  | Precision macro | 0.993 ± 0.004 | 0.986 |
  | Recall macro | 0.995 ± 0.004 | 0.983 |

- Matriz de confusão (destaque nos 3 falsos negativos vs 103 verdadeiros positivos)
- Gráfico de feature importance (INDE, IAN, INDICE_BEMESTAR no topo)

---

## Bloco 5 — Encerramento e impacto
### ⏱ 4:15–5:00 · 45 segundos

**Objetivo:** Fechar com visão de impacto e próximos passos.

**Fala:**
> "O que entregamos vai além de um modelo preditivo.
>
> Entregamos uma plataforma: com API pública documentada que a própria
> equipe da Passos Mágicos pode integrar aos seus sistemas, monitoramento
> para garantir que o modelo continue confiável ao longo do tempo, e uma
> pipeline de retreinamento automático para incorporar dados novos a cada
> ciclo letivo.
>
> Como próximos passos, visualizamos a integração com o sistema interno
> da associação e a expansão do modelo para prever não só defasagem, mas
> o potencial de ponto de virada de cada estudante.
>
> A tecnologia aqui é o meio. O fim é garantir que nenhuma criança da
> Passos Mágicos seja deixada para trás."

**Na tela:**
- URL pública da API na Render (ex: `https://passos-magicos-api.onrender.com/docs`)
- QR code para o repositório GitHub
- Slide final com nome do grupo e logo Passos Mágicos

---

## Referência Rápida — Métricas Reais do Modelo

> Treinado em: **09/03/2026** · Algoritmo: **LightGBM** · Versão: **1.0.0**

| Métrica | Valor |
|---------|-------|
| F1-Score macro (teste) | **0.984** |
| ROC-AUC (teste) | **0.999** |
| Precision macro (teste) | 0.986 |
| Recall macro (teste) | 0.983 |
| F1-Score macro (CV 5-fold) | 0.994 ± 0.002 |
| Acurácia (teste) | 0.989 |
| Falsos negativos (alunos em risco não detectados) | **3 de 106** |
| Total de amostras de teste | 455 |
| Features utilizadas | 18 |

---

## Dicas de Gravação

### Formato e qualidade
- Resolução mínima **1080p**
- Iluminação frontal, sem sombras no rosto
- **Microfone externo** se possível — áudio ruim elimina qualidade visual
- Fundo neutro ou com logo do projeto visível

### O que NÃO fazer
- ❌ Não leia o código linha por linha — mostre funcionando
- ❌ Não gaste mais de 30 segundos explicando a estrutura de pastas
- ❌ Não omita a demo ao vivo do `/predict` — é o ponto mais convincente
- ❌ Não termine sem mostrar a URL pública da API na Render
- ❌ Não use linguagem excessivamente técnica nos blocos 1, 2 e 5

### O que FAZER
- ✅ Abra com o problema humano antes de qualquer tecnologia
- ✅ Mostre o GitHub Actions verde — é prova de qualidade de processo
- ✅ Tenha F1 = **0.984** e ROC-AUC = **0.999** na ponta da língua
- ✅ Mencione "produção", "confiabilidade" e "monitoramento"
- ✅ Feche com impacto social, não com detalhes técnicos
- ✅ Ensaie ao menos 2 vezes antes de gravar — 5 minutos é apertado

---

*Roteiro gerado automaticamente com métricas reais de `app/model/metadata.joblib`.*
