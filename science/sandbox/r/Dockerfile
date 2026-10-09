ARG R_BASE_IMAGE
FROM ${R_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends r-cran-jsonlite \
    && rm -rf /var/lib/apt/lists/*
