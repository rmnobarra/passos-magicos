# Comando: /test

Executar suíte completa de testes unitários com relatório de cobertura.

## Instruções

1. **Verificar instalação**: confirmar que `pytest` e `pytest-cov` estão instalados

2. **Executar testes**:
```bash
pytest tests/ \
  --cov=src \
  --cov=app \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-fail-under=80 \
  -v
```

3. **Analisar resultado**: 
   - Se cobertura < 80%: identificar módulos com baixa cobertura e adicionar testes faltantes
   - Se testes falhando: investigar e corrigir antes de prosseguir

4. **Relatório**: indicar quais arquivos estão com cobertura abaixo do mínimo e sugerir testes adicionais

## Saída esperada
```
PASSED tests/test_preprocessing.py::test_handle_missing_values
PASSED tests/test_api.py::test_predict_returns_200
...
Coverage: 85% ✅
```
