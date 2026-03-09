# Comando: /monitor

Iniciar ou verificar o sistema de monitoramento de drift do modelo.

## Instruções

1. **Verificar dependências**: `evidently` e `streamlit` instalados

2. **Gerar relatório de drift** (comparar dados de referência vs dados recentes):
```bash
python monitoring/drift_detector.py
```

3. **Subir dashboard** (se não estiver rodando via docker-compose):
```bash
streamlit run monitoring/dashboard.py --server.port 8501
```

4. **Verificar logs de predições**:
```bash
tail -f logs/api.log
```

5. **Analisar alertas**: se drift detectado (>15%), reportar:
   - Quais features drifaram
   - Score de drift
   - Recomendar retreinamento se score > 30%

## Saída esperada
```
📊 Dashboard disponível em: http://localhost:8501
📋 Último relatório: reports/drift_report_YYYYMMDD_HHMM.html
⚠️  Status: SEM DRIFT DETECTADO (score: 8%)
```
