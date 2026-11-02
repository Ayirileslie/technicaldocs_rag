import socket

from techdocs_rag.config import Settings

raw = Settings.from_env().weaviate_url
print("WEAVIATE_URL repr:", repr(raw))
host = raw.replace("https://", "").replace("http://", "").strip().strip("\"'").strip("/")
for h in (host, "grpc-" + host):
    try:
        print(h, "->", sorted({a[4][0] for a in socket.getaddrinfo(h, 443)}))
    except Exception as e:
        print(h, "-> FAILED:", e)