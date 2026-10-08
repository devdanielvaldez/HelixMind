FROM rust:1.85-bookworm AS builder
WORKDIR /src
COPY Cargo.toml ./
COPY crates ./crates
RUN cargo build --release -p medai-api -p medai

FROM debian:bookworm-slim AS api
RUN useradd --system --uid 10001 medai && mkdir -p /data && chown medai:medai /data
COPY --from=builder /src/target/release/medai-api /usr/local/bin/medai-api
COPY --from=builder /src/target/release/medai /usr/local/bin/medai
USER 10001
ENV MEDAI_BIND=0.0.0.0:8090 MEDAI_DB=/data/medai.sqlite
EXPOSE 8090
ENTRYPOINT ["/usr/local/bin/medai-api"]
