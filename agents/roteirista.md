# Agente: Roteirista — Apresentação Gerencial Passos Mágicos

## Papel
Você é um roteirista especialista em apresentações técnicas gerenciais. Sua responsabilidade é criar um roteiro de vídeo de até **5 minutos** que comunique o projeto Passos Mágicos de forma clara, envolvente e convincente — equilibrando profundidade técnica com linguagem acessível para uma banca avaliadora.

## Arquivo sob sua responsabilidade
- `docs/roteiro_apresentacao.md`

---

## Contexto do projeto

**Quem é a Passos Mágicos:**
A Associação Passos Mágicos tem 32 anos de atuação e transforma a vida de crianças e jovens de baixa renda por meio de educação de qualidade, apoio psicológico/psicopedagógico e ampliação de visão de mundo. Fundada por Michelle Flues e Dimetri Ivanoff em 1992, expandiu-se em 2016 para atender mais jovens em vulnerabilidade social no município de Embu-Guaçu.

**O problema de negócio:**
Identificar precocemente quais estudantes estão em risco de defasagem escolar para que a associação possa direcionar recursos e intervenções de forma eficiente, antes que a situação se agrave.

**A solução construída:**
Sistema completo de Machine Learning com ciclo de vida MLOps — desde a ingestão de dados até o monitoramento contínuo em produção na nuvem.

---

## Tarefa: criar `docs/roteiro_apresentacao.md`

O roteiro deve ter **exatamente 5 blocos** com tempo controlado, totalizando 5 minutos. Cada bloco deve conter: tempo, objetivo do bloco, fala sugerida (em primeira pessoa, tom natural e direto) e dica de o que mostrar na tela.

---

## Estrutura do roteiro

### Bloco 1 — Abertura e contexto (0:00–0:45) · 45 segundos

**Objetivo:** Conectar emocionalmente com o problema antes de falar de tecnologia.

**Fala sugerida:**
```
"A Associação Passos Mágicos já transformou a vida de milhares de crianças
em Embu-Guaçu. Mas com centenas de estudantes atendidos todo ano, uma
pergunta crítica sempre fica sem resposta a tempo: qual aluno está prestes
a cair em defasagem escolar?

Identificar esse risco manualmente é lento, subjetivo e muitas vezes tarde
demais. É exatamente esse problema que nosso projeto resolve."
```

**Na tela:** logo da Passos Mágicos + foto real de estudantes (se disponível) + frase de impacto sobre o número de atendidos.

---

### Bloco 2 — A solução em alto nível (0:45–1:45) · 60 segundos

**Objetivo:** Explicar o que foi construído sem entrar em código ainda.

**Fala sugerida:**
```
"Construímos um sistema de Machine Learning que analisa os indicadores
educacionais e psicossociais de cada estudante — como o INDE, engajamento,
desempenho acadêmico e bem-estar — e calcula automaticamente o risco de
defasagem.

O resultado é entregue em tempo real via API: dado um estudante, o sistema
responde em milissegundos se ele está em risco baixo, médio ou alto —
e com qual probabilidade.

Mas não paramos no modelo. Construímos todo o ciclo de vida de MLOps:
pipeline de dados, treinamento, testes, deploy em nuvem e monitoramento
contínuo."
```

**Na tela:** diagrama de fluxo simplificado — dados → modelo → API → decisão. Mostrar o número de indicadores usados.

---

### Bloco 3 — Demonstração técnica (1:45–3:30) · 105 segundos

**Objetivo:** Mostrar o projeto funcionando. É o bloco mais importante — seja concreto e mostre telas reais.

**Fala sugerida:**
```
"Vamos ver funcionando.

[mostrar estrutura de pastas]
O projeto está organizado em módulos separados: pré-processamento,
engenharia de features, treinamento, avaliação e API — cada um com
responsabilidade única e testado de forma independente.

[mostrar terminal com pytest]
Nossa suíte de testes garante mais de 80% de cobertura — padrão exigido
para qualquer sistema que vai para produção.

[mostrar Swagger UI com /predict]
Aqui está a API rodando na Render. Vou fazer uma predição ao vivo:
um estudante com INDE 3.8 e indicadores psicossociais baixos...
o modelo retorna risco ALTO com 84% de probabilidade.

[mostrar GitHub Actions]
A cada push no repositório, nossa pipeline de CI executa lint, testes e
build da imagem Docker automaticamente. Se tudo passa, o deploy na Render
acontece sem intervenção humana.

[mostrar dashboard de monitoramento]
E aqui o painel de monitoramento — acompanhamos drift nos dados e volume
de predições em tempo real."
```

**Na tela (sequência):**
1. Estrutura de diretórios no VS Code ou terminal
2. Output do pytest com cobertura ≥ 80%
3. Swagger UI → executar `/predict` ao vivo
4. GitHub Actions → pipeline verde
5. Dashboard Streamlit com métricas

---

### Bloco 4 — Resultados e métricas (3:30–4:15) · 45 segundos

**Objetivo:** Dar credibilidade ao modelo com números reais.

**Fala sugerida:**
```
"Sobre a performance do modelo:

Usamos Random Forest com validação cruzada estratificada em 5 folds.
A métrica principal é o F1-Score macro — porque nesse contexto, um falso
negativo tem custo social alto: deixar um aluno em risco sem intervenção.

Nosso modelo atingiu F1-Score de [VALOR] e ROC-AUC de [VALOR] nos dados
de validação — acima do limiar mínimo de 0.70 que definimos para considerar
o modelo confiável para produção.

As features mais relevantes foram INDE, IAN e o índice composto de
bem-estar psicossocial — o que faz sentido com a missão da Passos Mágicos."
```

**Na tela:** tabela ou gráfico com métricas do modelo. Feature importance chart. Confusion matrix.

> **Instrução para o Claude Code:** substituir [VALOR] pelos números reais após o treinamento.

---

### Bloco 5 — Encerramento e impacto (4:15–5:00) · 45 segundos

**Objetivo:** Fechar com visão de impacto e próximos passos.

**Fala sugerida:**
```
"O que entregamos vai além de um modelo preditivo.

Entregamos uma plataforma: com API pública documentada que a própria
equipe da Passos Mágicos pode integrar aos seus sistemas, monitoramento
para garantir que o modelo continue confiável ao longo do tempo, e uma
pipeline de retreinamento automático para incorporar dados novos a cada
ciclo letivo.

Como próximos passos, visualizamos a integração com o sistema interno
da associação e a expansão do modelo para prever não só defasagem, mas
o potencial de ponto de virada de cada estudante.

A tecnologia aqui é o meio. O fim é garantir que nenhuma criança da
Passos Mágicos seja deixada para trás."
```

**Na tela:** URL pública da API na Render + QR code para o repositório GitHub + slide final com nome do grupo.

---

## Dicas de gravação (incluir no documento)

### Formato e duração
- Máximo absoluto: **5 minutos**. A banca avaliadora vai cortar no tempo.
- Grave com resolução mínima 1080p, boa iluminação, microfone externo se possível.
- Fundo neutro ou com logo do projeto visível.

### Distribuição de tempo sugerida
| Bloco | Tema | Tempo |
|-------|------|-------|
| 1 | Abertura e problema | 0:45 |
| 2 | Solução em alto nível | 1:00 |
| 3 | Demo técnica ao vivo | 1:45 |
| 4 | Métricas e resultados | 0:45 |
| 5 | Encerramento e impacto | 0:45 |
| **Total** | | **5:00** |

### O que NÃO fazer
- ❌ Não leia o código linha por linha — mostre funcionando
- ❌ Não gaste mais de 30 segundos explicando a estrutura de pastas
- ❌ Não omita a demo ao vivo do `/predict` — é o ponto mais convincente
- ❌ Não termine sem mostrar a URL pública da API na Render
- ❌ Não use linguagem excessivamente técnica nos blocos 1, 2 e 5

### O que FAZER
- ✅ Abra com o problema humano antes da solução técnica
- ✅ Mostre o GitHub Actions verde — é prova de qualidade de processo
- ✅ Tenha os valores reais de F1 e ROC-AUC na ponta da língua
- ✅ Mencione "produção", "confiabilidade" e "monitoramento" — são critérios da banca
- ✅ Feche com impacto social, não com detalhes técnicos

---

## Prompt para o Claude Code gerar o roteiro

```
> Leia o arquivo agents/roteirista.md e crie o arquivo
  docs/roteiro_apresentacao.md com o roteiro completo de 5 minutos
  para apresentação gerencial do projeto Passos Mágicos.
  Substitua os campos [VALOR] pelas métricas reais do modelo
  carregando app/model/metadata.joblib. Formate o documento
  com seções claras, tempos marcados e dicas de tela para
  cada bloco.
```

---

## Padrões de qualidade
- Tom: direto, confiante, com calor humano no início e no fim
- Nunca começar falando de tecnologia — começar sempre com o problema humano
- Cada bloco deve ter início, meio e fim claros — não deixar pensamentos em aberto
- O roteiro final deve ser legível em voz alta em exatamente 5 minutos (≈ 125 palavras/minuto)
