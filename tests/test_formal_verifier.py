"""Unit tests for SMT Formal Verification and Theorem Proving."""

from __future__ import annotations

from typing import Any

from security.formal_verifier import FormalStateVerifier, VerificationStatus


def test_smt_routing_loop_proof_and_counter_example() -> None:
    """Prove acyclic valid routes vs detecting cyclic routing loop counter-examples."""
    verifier = FormalStateVerifier(use_z3=True)

    # 1. Valid routing table
    valid_routes = [
        {"destination": "default", "gateway": "192.0.2.1", "interface": "eth0"},
        {"destination": "10.0.0.0/8", "gateway": "192.0.2.254", "interface": "eth1"},
    ]
    res_valid = verifier.verify_routing_table(valid_routes)
    assert res_valid.proved_safe is True
    assert res_valid.status == VerificationStatus.PROVED_SAFE
    assert res_valid.counter_example is None

    # 2. Invalid routing cycle: A -> B -> A
    cyclic_routes = [
        {"destination": "10.0.0.0/24", "gateway": "10.0.1.1"},
        {"destination": "10.0.1.1", "gateway": "10.0.0.0/24"},
    ]
    res_cycle = verifier.verify_routing_table(cyclic_routes)
    assert res_cycle.proved_safe is False
    assert res_cycle.status == VerificationStatus.COUNTER_EXAMPLE_FOUND
    assert res_cycle.counter_example is not None


def test_smt_firewall_shadow_rule_detection() -> None:
    """Verify detection of shadowed rules and blocked management ports."""
    verifier = FormalStateVerifier(use_z3=True)

    # 1. Valid firewall rules
    valid_rules = [
        "-A INPUT -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT",
        "-A INPUT -p tcp --dport 22 -j ACCEPT",
        "-A INPUT -p udp --dport 53 -j ACCEPT",
        "-A INPUT -p tcp --dport 9100 -j ACCEPT",
        "-A INPUT -j DROP",
    ]
    res_valid = verifier.verify_firewall_rules(valid_rules)
    assert res_valid.proved_safe is True
    assert res_valid.status == VerificationStatus.PROVED_SAFE

    # 2. Shadowed rule: ACCEPT after global DROP
    shadowed_rules = [
        "-A INPUT -p all -j DROP",
        "-A INPUT -p tcp --dport 80 -j ACCEPT",
    ]
    res_shadow = verifier.verify_firewall_rules(shadowed_rules)
    assert res_shadow.proved_safe is False
    assert res_shadow.status == VerificationStatus.COUNTER_EXAMPLE_FOUND
    assert "shadow" in res_shadow.proof_trace.lower()


def test_smt_permission_boundary_security_proof() -> None:
    """Prove security bounds on file permission bit lattices."""
    verifier = FormalStateVerifier(use_z3=True)

    # 1. Safe least-privilege permissions
    safe_perms = {
        "/etc/sudoers": "0440",
        "/etc/shadow": "0600",
        "/root/.ssh/id_rsa": "0600",
        "/usr/bin/python3": "0755",
    }
    res_safe = verifier.verify_permission_boundaries(safe_perms)
    assert res_safe.proved_safe is True
    assert res_safe.status == VerificationStatus.PROVED_SAFE

    # 2. Insecure world-writable sudoers violation
    insecure_perms = {
        "/etc/sudoers": "0777",
    }
    res_insecure = verifier.verify_permission_boundaries(insecure_perms)
    assert res_insecure.proved_safe is False
    assert res_insecure.status == VerificationStatus.COUNTER_EXAMPLE_FOUND
    assert res_insecure.counter_example is not None
    assert "/etc/sudoers" in res_insecure.counter_example["path"]


def test_smt_remediation_diff_combined_verification() -> None:
    """Test full multi-domain remediation diff formal proof."""
    verifier = FormalStateVerifier(use_z3=True)

    pre_state: dict[str, Any] = {}
    post_state = {
        "routes": [{"destination": "default", "gateway": "192.0.2.1", "interface": "eth0"}],
        "firewall_rules": [
            "-A INPUT -p tcp --dport 22 -j ACCEPT",
            "-A INPUT -p tcp --dport 9100 -j ACCEPT",
            "-A INPUT -p udp --dport 53 -j ACCEPT",
        ],
        "file_permissions": {
            "/etc/sudoers": "0440",
            "/etc/resolv.conf": "0644",
        },
    }

    res = verifier.verify_remediation_diff(pre_state, post_state)
    assert res.proved_safe is True
    assert res.domain == "combined"
    assert len(res.invariants_checked) == 3
    assert res.duration_ms >= 0.0
