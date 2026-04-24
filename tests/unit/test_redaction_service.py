"""Redaction tests. Plan §3.2 / §7.2 / §23."""

from __future__ import annotations

from siqueira_memo.services.redaction_service import RedactionService, redact


def test_redacts_openai_style_key():
    text = "Here is my key sk-proj-abcdef0123456789abcdef0123456789abcdef0123456789abcd"
    result = redact(text)
    assert "sk-proj-abcdef" not in result.redacted
    assert "[SECRET_REF:" in result.redacted
    assert result.matches >= 1
    assert any(m.kind == "openai_api_key" for m in result.findings)


def test_redacts_anthropic_key():
    text = "ANTHROPIC_API_KEY=sk-ant-api03-ABCDEFGHIJKLMNOPqrstuvwxyz0123456789ABCDEFGHIJKLMNOPqrstuvwxyz0123456789ABCDEFGHIJK-AA"
    result = redact(text)
    assert "sk-ant-" not in result.redacted
    assert "[SECRET_REF:" in result.redacted


def test_redacts_bearer_token():
    text = "Authorization: Bearer abcd1234EFGH5678ijkl9012mnop3456qrst7890"
    result = redact(text)
    assert "abcd1234EFGH" not in result.redacted


def test_redacts_ssh_private_key_block():
    text = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZWQyNTUxOQAAACDQm/+BBs\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    result = redact(text)
    assert "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU" not in result.redacted
    assert "[SECRET_REF:ssh_private_key" in result.redacted


def test_redacts_pem_private_key_block():
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAq7C9kZGCxMNoz+7dI1pYTkWEABHBkwBkRlj+vHqScURrH1ySaRT\n"
        "-----END RSA PRIVATE KEY-----"
    )
    result = redact(text)
    assert "MIIEowIBAAKCAQEA" not in result.redacted
    assert "pem_private_key" in result.redacted


def test_redacts_jwt():
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    text = f"token={jwt}"
    result = redact(text)
    assert jwt not in result.redacted
    assert "jwt" in result.redacted


def test_redacts_database_url():
    text = "DATABASE_URL=postgres://user:p4ssw0rd!@db.example.com:5432/mydb"
    result = redact(text)
    # The password portion must be scrubbed; host/db can survive as SECRET_REF.
    assert "p4ssw0rd" not in result.redacted


def test_redacts_env_block():
    text = (
        "Contents of .env:\n"
        "OPENAI_API_KEY=sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1\n"
        "SECRET_TOKEN=super-secret-nobody-knows\n"
    )
    result = redact(text)
    assert "sk-proj-aaaa" not in result.redacted
    assert "super-secret" not in result.redacted


def test_redacts_seed_phrase():
    text = (
        "My recovery phrase is: "
        "unveil actual spring scale fence slogan verify merit cloth rose quality detail"
    )
    result = redact(text)
    assert "unveil actual spring scale fence slogan" not in result.redacted


def test_redacts_telegram_bot_token():
    text = "TG bot token: 123456789:AAHEXAMPLEtokenAAAAAAAAAAAAAAAAAAAAa"
    result = redact(text)
    assert "AAHEXAMPLEtoken" not in result.redacted


def test_redacts_github_token():
    text = "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789AB"
    result = redact(text)
    assert "ghp_abcdef" not in result.redacted


def test_false_positives_preserve_public_identifiers():
    text = "Please push to github.com/acme/repo and check wallet 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEB0."
    result = redact(text)
    # We do not destroy public identifiers (plan §23.3).
    assert "github.com/acme/repo" in result.redacted
    assert "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEB0" in result.redacted


def test_preserves_normal_conversation():
    text = "Мы договорились использовать Hermes MemoryProvider plugin как primary."
    result = redact(text)
    assert result.redacted == text
    assert result.matches == 0


def test_redact_corpus_recall_above_threshold(tmp_path):
    # plan §23.4: known secret recall >= 99% on test corpus.
    corpus = [
        ("sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "openai_api_key"),
        ("sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AA", "anthropic_api_key"),
        ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789AB", "github_token"),
        ("Bearer abcd1234EFGH5678ijkl9012mnop3456qrst7890", "bearer_token"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", "jwt"),
        ("postgres://user:Pa55word@db.example.com:5432/mydb", "database_url"),
        ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_access_key"),
        ("AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
        ("123456789:AAH4exAMPLEtokenAAAAAAAAAAAAAAAAAAAAA", "telegram_bot_token"),
        ("unveil actual spring scale fence slogan verify merit cloth rose quality detail", "seed_phrase"),
    ]
    hits = 0
    for secret, _kind in corpus:
        result = redact(f"prefix {secret} suffix")
        if secret not in result.redacted:
            hits += 1
    assert hits / len(corpus) >= 0.99


def test_false_positive_rate_below_threshold():
    public = [
        "github.com/torvalds/linux",
        "wallet 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEB0",
        "server 192.168.1.42",
        "domain example.com",
        "Mark spoke about Shannon auth design",
        "https://github.com/siqueira/memo",
        "OpenAI documentation at platform.openai.com",
        "email: mark@example.com",
        "path: /home/mark/code/project",
        "Hermes MemoryProvider plugin",
    ]
    fp = 0
    for text in public:
        result = redact(text)
        if result.matches > 0:
            fp += 1
    assert fp / len(public) <= 0.2


def test_redaction_service_counts_per_kind():
    svc = RedactionService()
    text = (
        "sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa and "
        "Bearer abcd1234EFGH5678ijkl9012mnop3456qrst7890"
    )
    result = svc.redact(text)
    kinds = {m.kind for m in result.findings}
    assert "openai_api_key" in kinds
    assert "bearer_token" in kinds


def test_redaction_service_is_idempotent_on_placeholder():
    text = "[SECRET_REF:openai_api_key/unknown/deadbeef]"
    result = redact(text)
    # Already redacted text should not re-match.
    assert result.matches == 0
    assert result.redacted == text
