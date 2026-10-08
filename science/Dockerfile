FROM python:3.12-slim
WORKDIR /app
COPY science/pyproject.toml ./pyproject.toml
COPY science/src ./src
RUN pip install --no-cache-dir -e ".[mcp]"
RUN useradd --system --uid 10001 medai && mkdir -p /data && chown -R medai:medai /data
USER 10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV MEDAI_PROVENANCE_KEY_FILE=/data/keys/provenance-hmac.key
ENV MEDAI_MCP_HOST=0.0.0.0 MEDAI_MCP_PORT=8091 MEDAI_MCP_TRANSPORT=streamable-http
EXPOSE 8091
ENTRYPOINT ["medai-mcp"]
