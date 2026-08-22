from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict


@dataclass
class EvidenceRecord:
    claim_id: str
    customer_id: str
    image_hash: str = ""
    evidence_signature: str = ""
    description: str = ""


class EvidenceReuseDetector:

    def __init__(self):
        self.records: List[EvidenceRecord] = []

        self.image_index = defaultdict(list)
        self.signature_index = defaultdict(list)

    def add_evidence(
        self,
        claim_id,
        customer_id,
        image_hash="",
        evidence_signature="",
        description="",
    ):

        record = EvidenceRecord(
            claim_id=claim_id,
            customer_id=customer_id,
            image_hash=image_hash,
            evidence_signature=evidence_signature,
            description=description,
        )

        self.records.append(record)

        if image_hash:
            self.image_index[image_hash].append(record)

        if evidence_signature:
            self.signature_index[evidence_signature].append(record)

    def analyze_claim(self, claim_id):

        target = None

        for record in self.records:
            if record.claim_id == claim_id:
                target = record
                break

        if target is None:
            raise ValueError(
                f"Claim '{claim_id}' does not exist."
            )

        reused_image_claims = []
        reused_signature_claims = []

        if target.image_hash:
            for record in self.image_index.get(
                target.image_hash, []
            ):
                if record.claim_id != claim_id:
                    reused_image_claims.append(
                        record.claim_id
                    )

        if target.evidence_signature:
            for record in self.signature_index.get(
                target.evidence_signature, []
            ):
                if record.claim_id != claim_id:
                    reused_signature_claims.append(
                        record.claim_id
                    )

        reused_image_claims = sorted(
            set(reused_image_claims)
        )

        reused_signature_claims = sorted(
            set(reused_signature_claims)
        )

        image_reuse = len(reused_image_claims)
        signature_reuse = len(reused_signature_claims)

        score = 0.0

        if image_reuse:
            score += min(0.60, 0.30 * image_reuse)

        if signature_reuse:
            score += min(0.40, 0.20 * signature_reuse)

        score = min(score, 1.0)

        if score >= 0.70:
            status = "HIGH_REUSE"
            recommendation = "EVIDENCE_FRAUD_INVESTIGATION"

        elif score >= 0.30:
            status = "POSSIBLE_REUSE"
            recommendation = "EVIDENCE_REVIEW"

        elif score > 0:
            status = "LOW_REUSE"
            recommendation = "MONITOR"

        else:
            status = "NO_REUSE"
            recommendation = "NO_EVIDENCE_ESCALATION"

        reasons = []

        if image_reuse:
            reasons.append(
                f"Evidence image reused across "
                f"{image_reuse} previous claim(s)."
            )

        if signature_reuse:
            reasons.append(
                f"Evidence signature matches "
                f"{signature_reuse} previous claim(s)."
            )

        if not reasons:
            reasons.append(
                "No evidence reuse detected."
            )

        return {
            "claim_id": claim_id,
            "image_reuse_count": image_reuse,
            "signature_reuse_count": signature_reuse,
            "reused_image_claims": reused_image_claims,
            "reused_signature_claims": reused_signature_claims,
            "reuse_score": round(score, 4),
            "status": status,
            "recommendation": recommendation,
            "reasons": reasons,
        }


if __name__ == "__main__":

    detector = EvidenceReuseDetector()

    detector.add_evidence(
        "CLM001",
        "C001",
        image_hash="IMG_ABC",
        evidence_signature="DAMAGE_HEADPHONE",
        description="Broken headphone headband",
    )

    detector.add_evidence(
        "CLM002",
        "C002",
        image_hash="IMG_ABC",
        evidence_signature="DAMAGE_HEADPHONE",
        description="Broken headphone headband",
    )

    detector.add_evidence(
        "CLM003",
        "C003",
        image_hash="IMG_XYZ",
        evidence_signature="DAMAGE_SCREEN",
        description="Cracked phone screen",
    )

    result = detector.analyze_claim("CLM002")

    print("=" * 70)
    print("TRUSTLOOP — EVIDENCE REUSE DETECTOR")
    print("=" * 70)

    print()
    print(f"Claim ID             : {result['claim_id']}")
    print(
        f"Image reuse count    : "
        f"{result['image_reuse_count']}"
    )
    print(
        f"Signature reuse count: "
        f"{result['signature_reuse_count']}"
    )
    print(
        f"Reuse score          : "
        f"{result['reuse_score']:.2%}"
    )
    print(
        f"Status               : "
        f"{result['status']}"
    )
    print(
        f"Recommendation       : "
        f"{result['recommendation']}"
    )

    print()
    print("Reasons:")

    for reason in result["reasons"]:
        print(f"  - {reason}")

    print()
    print("=" * 70)
    print("EVIDENCE REUSE TEST COMPLETED")
    print("=" * 70)
