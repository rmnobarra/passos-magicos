# Comando: /train

Treinar o modelo de predição de risco de defasagem escolar do projeto Passos Mágicos.

## Instruções

Execute o seguinte fluxo completo de treinamento:

1. **Verificar dataset**: confirmar que existe arquivo CSV em `data/raw/`. Se não existir, criar dados sintéticos de desenvolvimento em `data/raw/synthetic_data.csv` com o schema correto (colunas: NOME, ANO, FASE, TURMA, INDE, IAN, IDA, IEG, IAA, IPS, IPP, IPV, PONTO_DE_VIRADA, DEFASAGEM)

2. **Executar preprocessing**: `python -c "from src.preprocessing import DataPreprocessor; ..."`

3. **Executar feature engineering**: aplicar `FeatureEngineer.transform()`

4. **Treinar modelo**: `python src/train.py`

5. **Avaliar modelo**: verificar se métricas atendem critério mínimo (F1 ≥ 0.70)

6. **Confirmar artefatos**: verificar existência de `app/model/model.joblib` e `app/model/metadata.joblib`

7. **Reportar resultado**: exibir tabela com métricas obtidas no treino e validação cruzada

## Saída esperada
```
✅ Modelo treinado com sucesso!
   F1-Score Macro (CV):  0.XX ± 0.XX
   ROC-AUC (CV):         0.XX ± 0.XX
   Recall Risco (CV):    0.XX ± 0.XX
   Artefato salvo em:    app/model/model.joblib
```
