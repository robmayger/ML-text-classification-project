# Dockerfile
FROM python:3.13-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl gnupg2 ca-certificates lsb-release

# Install code-server (VS Code in a browser)
RUN curl -fsSL https://code-server.dev/install.sh | sh

WORKDIR /code

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

# Expose the default code-server port
EXPOSE 8080

# Start code-server
CMD ["code-server", "--host", "0.0.0.0", "--port", "8080", "/code"]