"""Kalshi API request signing (https://docs.kalshi.com/getting_started/api_keys).

Signature = base64(sign(f"{timestamp_ms}{METHOD}{path_without_query}")), where the path
starts at the API root (e.g. ``/trade-api/v2/portfolio/balance``). Ed25519 keys sign the
message directly; RSA keys use RSA-PSS with SHA-256 (MGF1-SHA256, salt = digest length).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class KalshiSigner:
    def __init__(self, key_id: str, private_key: Ed25519PrivateKey | rsa.RSAPrivateKey) -> None:
        if not key_id:
            raise ValueError("Kalshi API key id is empty")
        if not isinstance(private_key, Ed25519PrivateKey | rsa.RSAPrivateKey):
            raise TypeError("Kalshi keys must be Ed25519 or RSA private keys")
        self._key_id = key_id
        self._private_key = private_key

    @classmethod
    def from_pem(cls, key_id: str, pem: bytes) -> KalshiSigner:
        key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(key, Ed25519PrivateKey | rsa.RSAPrivateKey):
            raise TypeError(f"Unsupported private key type: {type(key).__name__}")
        return cls(key_id, key)

    @classmethod
    def from_file(cls, key_id: str, path: Path) -> KalshiSigner:
        return cls.from_pem(key_id, path.read_bytes())

    @property
    def key_type(self) -> str:
        return "ed25519" if isinstance(self._private_key, Ed25519PrivateKey) else "rsa"

    def sign(self, message: str) -> str:
        data = message.encode("utf-8")
        if isinstance(self._private_key, Ed25519PrivateKey):
            signature = self._private_key.sign(data)
        else:
            signature = self._private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256(),
            )
        return base64.b64encode(signature).decode("ascii")

    def headers(self, method: str, path: str, timestamp_ms: int | None = None) -> dict[str, str]:
        ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        path_without_query = path.split("?", 1)[0]
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts + method.upper() + path_without_query),
        }

    def __repr__(self) -> str:  # never expose key material
        return f"KalshiSigner(key_type={self.key_type!r})"
