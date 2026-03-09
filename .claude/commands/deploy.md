# Comando: /deploy

Empacotar e subir a aplicação com Docker.

## Instruções

1. **Pré-verificação**:
   - Confirmar que `app/model/model.joblib` existe (treinar se necessário)
   - Confirmar que `Dockerfile` e `docker-compose.yml` existem

2. **Build da imagem**:
```bash
docker build -t passos-magicos-api:latest .
```

3. **Subir com Docker Compose**:
```bash
docker-compose up --build -d
```

4. **Verificar saúde**:
```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

5. **Testar endpoint de predição**:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"inde":4.2,"ian":3.8,"ida":4.5,"ieg":5.0,"iaa":6.0,"ips":5.5,"ipp":5.2,"ipv":4.8,"fase":2,"ano":2024,"ponto_de_virada":false}'
```

6. **Reportar URLs disponíveis**:
   - API: http://localhost:8000/docs
   - Monitor: http://localhost:8501

## Resolução de problemas comuns
- Porta 8000 ocupada: alterar `ports` no docker-compose.yml
- Modelo não encontrado: executar `/train` antes do deploy
- Erro de build: verificar `requirements.txt` e compatibilidade de versões
