"""Unit tests for Confidential VM Remote Attestation."""

from __future__ import annotations

from security.confidential_attestation import (
    AttestationReport,
    ConfidentialAttestor,
)


def test_attestation_report_generation_and_detection() -> None:
    """Test generating hardware attestation reports across detected TEE environments."""
    attestor = ConfidentialAttestor()
    report = attestor.generate_attestation_report(user_data="test-nonce-123")

    assert isinstance(report, AttestationReport)
    assert len(report.measurement_digest) == 96  # SHA-384 hex
    assert report.signature_valid is True
    assert len(report.report_data) == 64
    assert report.report_data == "7a9c2b4a6171f03ed9f403889969421080fe4cc08f1b774eed9ee58e6a5b572b"


def test_attestation_verification_success_and_failure() -> None:
    """Test validating valid golden measurements vs detecting tampered launch digests."""
    attestor = ConfidentialAttestor()
    rep = attestor.generate_attestation_report()

    # 1. Matching measurement -> Verified
    res_valid = attestor.verify_attestation(rep, expected_measurement=rep.measurement_digest)
    assert res_valid.verified is True
    assert "verified" in res_valid.details.lower()

    # 2. Tampered measurement -> Refuted
    res_tampered = attestor.verify_attestation(rep, expected_measurement="0" * 96)
    assert res_tampered.verified is False
    assert "refuted" in res_tampered.details.lower()
