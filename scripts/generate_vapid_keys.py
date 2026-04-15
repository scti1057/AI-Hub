import base64
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def main():
    secrets_dir = Path("secrets")
    secrets_dir.mkdir(exist_ok=True)

    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    private_path = secrets_dir / "vapid_private.pem"
    private_path.write_bytes(private_pem)

    numbers = public_key.public_numbers()
    x = numbers.x.to_bytes(32, "big")
    y = numbers.y.to_bytes(32, "big")
    uncompressed_public_key = b"\x04" + x + y
    public_key_b64 = b64url(uncompressed_public_key)

    print("\nVAPID keys generated.\n")
    print(f"Private key file: {private_path}")
    print(f"VAPID public key:\n{public_key_b64}\n")
    print("Add these lines to your .env:")
    print(f"VAPID_PUBLIC_KEY={public_key_b64}")
    print(f"VAPID_PRIVATE_KEY_PATH={private_path}")
    print("VAPID_SUBJECT=mailto:DEINE_EMAIL")
    print()


if __name__ == "__main__":
    main()