"""
Gemma 4 API Gateway

로컬 Ollama 서버를 외부에 안전하게 노출합니다.
API Key 인증 + Rate Limiting + 엔드포인트 필터링.

사용법:
  python gateway/server.py                    # 기본 실행 (포트 8443)
  python gateway/server.py --port 9000        # 포트 지정
  python gateway/server.py key create "팀원A"  # API 키 생성
  python gateway/server.py key list           # 키 목록
  python gateway/server.py key revoke key_01  # 키 폐기
"""

import os
import sys
import json
import time
import hashlib
import secrets
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# pip install aiohttp
try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Install required: pip install aiohttp")
    sys.exit(1)

# 설정
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8443"))
KEYS_FILE = Path.home() / ".gemma4-agent" / "gateway" / "keys.json"

# 허용 엔드포인트
ALLOWED_ENDPOINTS = {
    "/api/chat",
    "/api/generate",
    "/api/show",
    "/api/tags",
    "/api/ps",
    "/api/embeddings",
    "/v1/chat/completions",
}

# 차단 엔드포인트
BLOCKED_ENDPOINTS = {
    "/api/delete",
    "/api/create",
    "/api/pull",
    "/api/push",
    "/api/copy",
}


# ============================================================
# API Key Management
# ============================================================

class KeyStore:
    def __init__(self):
        self.keys = {}
        self.load()

    def load(self):
        if KEYS_FILE.exists():
            data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
            self.keys = {k["id"]: k for k in data.get("keys", [])}

    def save(self):
        KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"keys": list(self.keys.values())}
        KEYS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, name, rate_limit=30):
        key_id = f"key_{len(self.keys) + 1:02d}"
        secret = f"gw-sk-{secrets.token_urlsafe(32)}"
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()

        self.keys[key_id] = {
            "id": key_id,
            "name": name,
            "secret_hash": secret_hash,
            "rate_limit": rate_limit,
            "enabled": True,
            "created_at": datetime.now().isoformat(),
        }
        self.save()
        return key_id, secret

    def validate(self, token):
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        for key in self.keys.values():
            if key["secret_hash"] == token_hash and key["enabled"]:
                return key
        return None

    def revoke(self, key_id):
        if key_id in self.keys:
            self.keys[key_id]["enabled"] = False
            self.save()
            return True
        return False

    def list_keys(self):
        return list(self.keys.values())


# ============================================================
# Rate Limiter
# ============================================================

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    def check(self, key_id, limit):
        now = time.time()
        window = 60  # 1분
        # 오래된 요청 제거
        self.requests[key_id] = [t for t in self.requests[key_id] if now - t < window]
        if len(self.requests[key_id]) >= limit:
            return False
        self.requests[key_id].append(now)
        return True


# ============================================================
# Gateway Server
# ============================================================

key_store = KeyStore()
rate_limiter = RateLimiter()


async def health_handler(request):
    return web.Response(text="OK")


async def proxy_handler(request):
    path = request.path

    # 엔드포인트 필터링
    if path in BLOCKED_ENDPOINTS:
        return web.json_response(
            {"error": f"Endpoint blocked: {path}"},
            status=403,
        )

    if path not in ALLOWED_ENDPOINTS and not path.startswith("/v1/") and path != "/health":
        return web.json_response(
            {"error": f"Endpoint not allowed: {path}"},
            status=403,
        )

    # 인증
    client_ip = request.remote or ""
    is_local = client_ip.startswith("127.") or client_ip.startswith("192.168.") \
        or client_ip.startswith("10.") or client_ip.startswith("172.") \
        or client_ip == "::1"

    auth = request.headers.get("Authorization", "")
    key_info = None

    if auth.startswith("Bearer "):
        # API Key가 있으면 검증
        token = auth[7:]
        key_info = key_store.validate(token)
        if not key_info:
            return web.json_response({"error": "Invalid API key"}, status=403)
    elif is_local:
        # 내부 네트워크는 인증 없이 허용
        key_info = {"id": f"local_{client_ip}", "rate_limit": 60}
    else:
        # 외부 IP는 API Key 필수
        return web.json_response(
            {"error": "Missing Authorization header. Use: Bearer <api-key>"},
            status=401,
        )

    # Rate limit
    if not rate_limiter.check(key_info["id"], key_info["rate_limit"]):
        return web.json_response(
            {"error": "Rate limit exceeded", "retry_after": 60},
            status=429,
        )

    # Ollama로 프록시
    target_url = f"{OLLAMA_URL}{path}"

    try:
        async with aiohttp.ClientSession() as session:
            # 요청 바디 읽기
            body = await request.read()

            # 헤더 복사 (인증 헤더 제외)
            headers = {}
            if request.content_type:
                headers["Content-Type"] = request.content_type

            # 스트리밍 여부 확인
            is_stream = False
            if body:
                try:
                    data = json.loads(body)
                    is_stream = data.get("stream", False)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            async with session.request(
                request.method,
                target_url,
                data=body,
                headers=headers,
            ) as resp:
                if is_stream:
                    # 스트리밍 응답
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={"Content-Type": resp.content_type or "application/x-ndjson"},
                    )
                    await response.prepare(request)
                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                    return response
                else:
                    # 일반 응답
                    response_body = await resp.read()
                    return web.Response(
                        body=response_body,
                        status=resp.status,
                        content_type=resp.content_type,
                    )

    except aiohttp.ClientError as e:
        return web.json_response(
            {"error": f"Ollama connection failed: {e}"},
            status=502,
        )


def create_app():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_route("*", "/{path:.*}", proxy_handler)
    return app


# ============================================================
# CLI
# ============================================================

def cli_key_create(name, rate_limit=30):
    key_id, secret = key_store.create(name, rate_limit)
    print(f"Created API key:")
    print(f"  ID:     {key_id}")
    print(f"  Name:   {name}")
    print(f"  Secret: {secret}")
    print(f"  Limit:  {rate_limit} req/min")
    print(f"\nStore this secret -- it cannot be retrieved later.")


def cli_key_list():
    keys = key_store.list_keys()
    if not keys:
        print("No keys. Create one: python gateway/server.py key create \"name\"")
        return
    print(f"{'ID':<10} {'Name':<20} {'Limit':<8} {'Status':<10}")
    print("-" * 50)
    for k in keys:
        status = "active" if k["enabled"] else "revoked"
        print(f"{k['id']:<10} {k['name']:<20} {k['rate_limit']:<8} {status:<10}")


def cli_key_revoke(key_id):
    if key_store.revoke(key_id):
        print(f"Revoked: {key_id}")
    else:
        print(f"Key not found: {key_id}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 API Gateway")
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start gateway server")
    serve_parser.add_argument("--port", type=int, default=GATEWAY_PORT)

    # key management
    key_parser = subparsers.add_parser("key", help="Manage API keys")
    key_sub = key_parser.add_subparsers(dest="action")

    create_p = key_sub.add_parser("create")
    create_p.add_argument("name", help="Key name")
    create_p.add_argument("--limit", type=int, default=30, help="Rate limit (req/min)")

    key_sub.add_parser("list")

    revoke_p = key_sub.add_parser("revoke")
    revoke_p.add_argument("id", help="Key ID to revoke")

    args = parser.parse_args()

    if args.command == "key":
        if args.action == "create":
            cli_key_create(args.name, args.limit)
        elif args.action == "list":
            cli_key_list()
        elif args.action == "revoke":
            cli_key_revoke(args.id)
        else:
            key_parser.print_help()
    elif args.command == "serve" or args.command is None:
        port = getattr(args, "port", GATEWAY_PORT)
        keys = key_store.list_keys()
        active = sum(1 for k in keys if k["enabled"])
        if active == 0:
            print("[WARN] No API keys! Create one:")
            print(f"  python gateway/server.py key create \"name\"")
            print()

        print(f"=========================================")
        print(f"  Gemma 4 API Gateway")
        print(f"  Port:   {port}")
        print(f"  Ollama: {OLLAMA_URL}")
        print(f"  Keys:   {active} active")
        print(f"=========================================")
        print(f"\nExternal usage:")
        print(f"  curl http://<IP>:{port}/api/tags -H 'Authorization: Bearer <key>'")
        print(f"\nClaude Code from another PC:")
        print(f"  $env:OLLAMA_HOST = 'http://<IP>:{port}'")
        print(f"  ollama launch claude --model gemma4:31b")
        print()

        app = create_app()
        web.run_app(app, port=port, print=lambda *a: None)


if __name__ == "__main__":
    main()
