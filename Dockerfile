FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn streamlit requests faiss-cpu numpy pymupdf pydantic python-multipart
COPY orchestrator.py frontend.py ./
RUN echo '#!/bin/bash\npython orchestrator.py & \nstreamlit run frontend.py --server.port=80 --server.address=0.0.0.0\n' > start.sh
RUN chmod +x start.sh
EXPOSE 80 5000
CMD ["./start.sh"]