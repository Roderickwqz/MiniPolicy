# import socket

# HOST = "127.0.0.1"
# PORT = 22006
# TIMEOUT = 3

# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.settimeout(TIMEOUT)

# try:
#     sock.connect((HOST, PORT))
#     print(f"✅ TCP CONNECT OK: {HOST}:{PORT} is reachable")
# except Exception as e:
#     print(f"❌ TCP CONNECT FAILED: {HOST}:{PORT}")
#     print(f"   Reason: {e}")
# finally:
#     sock.close()

import requests

endpoints = [
    "http://localhost:22006/v1/meta",
    "http://localhost:22006/v1/.well-known/openid-configuration",
]

for url in endpoints:
    print(f"\nTesting {url}")
    try:
        r = requests.get(url, timeout=3)
        print("  status:", r.status_code)
        print("  body:", r.text[:300])
    except Exception as e:
        print("  ❌ failed:", e)
