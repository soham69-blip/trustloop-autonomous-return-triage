from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Set, List


@dataclass
class CustomerNode:
    customer_id: str
    address: str = ""
    phone: str = ""
    device_id: str = ""
    payment_id: str = ""


class IdentityFraudGraphV2:

    # Individual identifier strengths.
    # These are heuristic prototype weights, NOT trained ML weights.
    SIGNAL_WEIGHTS = {
        "address": 0.08,
        "phone": 0.15,
        "device_id": 0.20,
        "payment_id": 0.25,
    }

    def __init__(self):
        self.customers: Dict[str, CustomerNode] = {}

        self.indexes = {
            "address": defaultdict(set),
            "phone": defaultdict(set),
            "device_id": defaultdict(set),
            "payment_id": defaultdict(set),
        }

    # ---------------------------------------------------------
    # CUSTOMER MANAGEMENT
    # ---------------------------------------------------------

    def add_customer(
        self,
        customer_id,
        address="",
        phone="",
        device_id="",
        payment_id="",
    ):

        # Remove old index references if customer already exists.
        if customer_id in self.customers:
            self._remove_customer_from_indexes(customer_id)

        node = CustomerNode(
            customer_id=customer_id,
            address=address,
            phone=phone,
            device_id=device_id,
            payment_id=payment_id,
        )

        self.customers[customer_id] = node

        values = {
            "address": address,
            "phone": phone,
            "device_id": device_id,
            "payment_id": payment_id,
        }

        for field_name, value in values.items():
            if value:
                self.indexes[field_name][value].add(customer_id)

    def _remove_customer_from_indexes(self, customer_id):

        node = self.customers[customer_id]

        values = {
            "address": node.address,
            "phone": node.phone,
            "device_id": node.device_id,
            "payment_id": node.payment_id,
        }

        for field_name, value in values.items():

            if not value:
                continue

            self.indexes[field_name][value].discard(customer_id)

            if not self.indexes[field_name][value]:
                del self.indexes[field_name][value]

    # ---------------------------------------------------------
    # DIRECT LINKS
    # ---------------------------------------------------------

    def _shared_signals(self, customer_a, customer_b):

        a = self.customers[customer_a]
        b = self.customers[customer_b]

        signals = []

        if a.address and a.address == b.address:
            signals.append("address")

        if a.phone and a.phone == b.phone:
            signals.append("phone")

        if a.device_id and a.device_id == b.device_id:
            signals.append("device_id")

        if a.payment_id and a.payment_id == b.payment_id:
            signals.append("payment_id")

        return signals

    def _direct_links(self, customer_id):

        if customer_id not in self.customers:
            return {}

        node = self.customers[customer_id]

        candidates = set()

        values = {
            "address": node.address,
            "phone": node.phone,
            "device_id": node.device_id,
            "payment_id": node.payment_id,
        }

        for field_name, value in values.items():

            if value:
                candidates.update(
                    self.indexes[field_name].get(value, set())
                )

        candidates.discard(customer_id)

        links = {}

        for other_id in candidates:

            signals = self._shared_signals(
                customer_id,
                other_id
            )

            if signals:
                links[other_id] = signals

        return links

    # ---------------------------------------------------------
    # MULTI-SIGNAL LINK SCORE
    # ---------------------------------------------------------

    def _pair_score(self, signals):

        if not signals:
            return 0.0

        base_score = sum(
            self.SIGNAL_WEIGHTS.get(signal, 0.0)
            for signal in signals
        )

        # Multiple independent identifiers pointing
        # to the same account are more suspicious than
        # simply adding individual weights.
        if len(signals) >= 2:
            base_score += 0.10

        if len(signals) >= 3:
            base_score += 0.10

        if len(signals) == 4:
            base_score += 0.10

        return min(base_score, 1.0)

    # ---------------------------------------------------------
    # CLUSTER DETECTION
    # ---------------------------------------------------------

    def get_cluster(self, customer_id):

        if customer_id not in self.customers:
            raise ValueError(
                f"Customer '{customer_id}' does not exist."
            )

        visited = set()
        queue = deque([customer_id])

        while queue:

            current = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            links = self._direct_links(current)

            for linked_id in links:

                if linked_id not in visited:
                    queue.append(linked_id)

        return sorted(visited)

    # ---------------------------------------------------------
    # CLUSTER DENSITY
    # ---------------------------------------------------------

    def _cluster_density(self, cluster):

        if len(cluster) <= 1:
            return 0.0

        possible_edges = (
            len(cluster) * (len(cluster) - 1)
        ) / 2

        actual_edges = 0

        for i in range(len(cluster)):

            for j in range(i + 1, len(cluster)):

                signals = self._shared_signals(
                    cluster[i],
                    cluster[j]
                )

                if signals:
                    actual_edges += 1

        if possible_edges == 0:
            return 0.0

        return actual_edges / possible_edges

    # ---------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------

    def analyze_customer(self, customer_id):

        if customer_id not in self.customers:
            raise ValueError(
                f"Customer '{customer_id}' does not exist."
            )

        node = self.customers[customer_id]

        direct_links = self._direct_links(customer_id)

        linked_customers = sorted(direct_links.keys())

        # -----------------------------------------------------
        # Shared identifier counts
        # -----------------------------------------------------

        def shared_count(field_name):

            value = getattr(node, field_name)

            if not value:
                return 0

            return max(
                0,
                len(
                    self.indexes[field_name].get(
                        value,
                        set()
                    )
                ) - 1
            )

        shared_address_count = shared_count("address")
        shared_phone_count = shared_count("phone")
        shared_device_count = shared_count("device_id")
        shared_payment_count = shared_count("payment_id")

        # -----------------------------------------------------
        # Direct pair scores
        # -----------------------------------------------------

        pair_scores = {}

        for linked_id, signals in direct_links.items():

            pair_scores[linked_id] = {
                "signals": signals,
                "score": round(
                    self._pair_score(signals),
                    4
                ),
            }

        strongest_link_score = max(
            [x["score"] for x in pair_scores.values()],
            default=0.0
        )

        # -----------------------------------------------------
        # Cluster
        # -----------------------------------------------------

        cluster = self.get_cluster(customer_id)

        cluster_size = len(cluster)

        cluster_density = self._cluster_density(cluster)

        # -----------------------------------------------------
        # Multi-signal relationships
        # -----------------------------------------------------

        multi_signal_links = {
            customer_id_: data
            for customer_id_, data in pair_scores.items()
            if len(data["signals"]) >= 2
        }

        strong_links = {
            customer_id_: data
            for customer_id_, data in pair_scores.items()
            if data["score"] >= 0.30
        }

        # -----------------------------------------------------
        # Fraud-ring score
        # -----------------------------------------------------

        ring_score = 0.0

        # Multiple linked accounts.
        if len(linked_customers) >= 2:
            ring_score += min(
                0.20,
                len(linked_customers) * 0.05
            )

        # Larger connected cluster.
        if cluster_size >= 3:
            ring_score += min(
                0.20,
                (cluster_size - 2) * 0.05
            )

        # Dense cluster.
        ring_score += min(
            0.20,
            cluster_density * 0.20
        )

        # Multiple identifiers shared with another account.
        if multi_signal_links:
            ring_score += 0.20

        # Strong device/payment relationships.
        if shared_device_count > 0:
            ring_score += min(
                0.10,
                shared_device_count * 0.05
            )

        if shared_payment_count > 0:
            ring_score += min(
                0.15,
                shared_payment_count * 0.075
            )

        ring_score = min(ring_score, 1.0)

        # -----------------------------------------------------
        # Identity link score
        # -----------------------------------------------------

        direct_score = (
            min(shared_address_count * 0.08, 0.25)
            + min(shared_phone_count * 0.15, 0.30)
            + min(shared_device_count * 0.20, 0.35)
            + min(shared_payment_count * 0.25, 0.40)
        )

        # Multi-signal boost.
        if multi_signal_links:
            direct_score += 0.10

        # Strongest pair relationship contributes additional evidence.
        direct_score = max(
            direct_score,
            strongest_link_score
        )

        identity_link_score = min(
            direct_score,
            1.0
        )

        # -----------------------------------------------------
        # Risk level
        # -----------------------------------------------------

        combined_score = max(
            identity_link_score,
            ring_score
        )

        if combined_score >= 0.70:
            risk_level = "HIGH"

        elif combined_score >= 0.40:
            risk_level = "MEDIUM"

        elif combined_score > 0:
            risk_level = "LOW"

        else:
            risk_level = "NONE"

        # -----------------------------------------------------
        # Investigation reasons
        # -----------------------------------------------------

        reasons = []

        if shared_address_count:
            reasons.append(
                f"Address shared with "
                f"{shared_address_count} other account(s)"
            )

        if shared_phone_count:
            reasons.append(
                f"Phone shared with "
                f"{shared_phone_count} other account(s)"
            )

        if shared_device_count:
            reasons.append(
                f"Device shared with "
                f"{shared_device_count} other account(s)"
            )

        if shared_payment_count:
            reasons.append(
                f"Payment identifier shared with "
                f"{shared_payment_count} other account(s)"
            )

        if multi_signal_links:
            reasons.append(
                "Multiple independent identifiers connect "
                "this customer to another account"
            )

        if cluster_size >= 3:
            reasons.append(
                f"Customer belongs to a connected cluster "
                f"of {cluster_size} accounts"
            )

        if cluster_density >= 0.50:
            reasons.append(
                "Connected account cluster has high link density"
            )

        if strong_links:
            reasons.append(
                f"{len(strong_links)} strong identity link(s) detected"
            )

        if not reasons:
            reasons.append(
                "No suspicious identity linkage detected"
            )

        # -----------------------------------------------------
        # Investigation recommendation
        # -----------------------------------------------------

        if ring_score >= 0.70:
            recommendation = "FRAUD_RING_INVESTIGATION"

        elif identity_link_score >= 0.50:
            recommendation = "IDENTITY_INVESTIGATION"

        elif identity_link_score >= 0.25:
            recommendation = "MONITOR"

        else:
            recommendation = "NO_IDENTITY_ESCALATION"

        return {
            "customer_id": customer_id,

            "shared_address_count":
                shared_address_count,

            "shared_phone_count":
                shared_phone_count,

            "shared_device_count":
                shared_device_count,

            "shared_payment_count":
                shared_payment_count,

            "linked_customer_count":
                len(linked_customers),

            "linked_customers":
                linked_customers,

            "identity_link_score":
                round(identity_link_score, 4),

            "strongest_link_score":
                round(strongest_link_score, 4),

            "multi_signal_link_count":
                len(multi_signal_links),

            "strong_link_count":
                len(strong_links),

            "cluster_size":
                cluster_size,

            "cluster_members":
                cluster,

            "cluster_density":
                round(cluster_density, 4),

            "fraud_ring_score":
                round(ring_score, 4),

            "risk_level":
                risk_level,

            "investigation_recommendation":
                recommendation,

            "reasons":
                reasons,

            "pair_relationships":
                pair_scores,
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    graph = IdentityFraudGraphV2()

    graph.add_customer(
        "C001",
        address="ADDR_100",
        phone="PHONE_001",
        device_id="DEVICE_001",
        payment_id="PAY_001",
    )

    graph.add_customer(
        "C002",
        address="ADDR_100",
        phone="PHONE_002",
        device_id="DEVICE_002",
        payment_id="PAY_002",
    )

    graph.add_customer(
        "C003",
        address="ADDR_200",
        phone="PHONE_003",
        device_id="DEVICE_001",
        payment_id="PAY_003",
    )

    graph.add_customer(
        "C004",
        address="ADDR_300",
        phone="PHONE_004",
        device_id="DEVICE_004",
        payment_id="PAY_001",
    )

    # Extra synthetic accounts to test
    # multi-signal and cluster detection.
    graph.add_customer(
        "C005",
        address="ADDR_500",
        phone="PHONE_005",
        device_id="DEVICE_RING",
        payment_id="PAY_RING",
    )

    graph.add_customer(
        "C006",
        address="ADDR_500",
        phone="PHONE_006",
        device_id="DEVICE_RING",
        payment_id="PAY_RING",
    )

    graph.add_customer(
        "C007",
        address="ADDR_500",
        phone="PHONE_007",
        device_id="DEVICE_RING",
        payment_id="PAY_RING",
    )

    print("=" * 75)
    print(
        "TRUSTLOOP — IDENTITY & ADDRESS FRAUD GRAPH V2 TEST"
    )
    print("=" * 75)

    for customer_id in [
        "C001",
        "C002",
        "C003",
        "C004",
        "C005",
        "C006",
        "C007",
    ]:

        result = graph.analyze_customer(customer_id)

        print()
        print(f"CUSTOMER: {customer_id}")
        print("-" * 75)

        print(
            f"Shared address          : "
            f"{result['shared_address_count']}"
        )

        print(
            f"Shared phone            : "
            f"{result['shared_phone_count']}"
        )

        print(
            f"Shared device           : "
            f"{result['shared_device_count']}"
        )

        print(
            f"Shared payment          : "
            f"{result['shared_payment_count']}"
        )

        print(
            f"Linked customers        : "
            f"{result['linked_customer_count']}"
        )

        print(
            f"Identity link score     : "
            f"{result['identity_link_score']:.2%}"
        )

        print(
            f"Strongest link score    : "
            f"{result['strongest_link_score']:.2%}"
        )

        print(
            f"Multi-signal links      : "
            f"{result['multi_signal_link_count']}"
        )

        print(
            f"Connected cluster size  : "
            f"{result['cluster_size']}"
        )

        print(
            f"Cluster density         : "
            f"{result['cluster_density']:.2%}"
        )

        print(
            f"Fraud-ring score        : "
            f"{result['fraud_ring_score']:.2%}"
        )

        print(
            f"Risk level              : "
            f"{result['risk_level']}"
        )

        print(
            f"Recommendation          : "
            f"{result['investigation_recommendation']}"
        )

        print(
            f"Cluster members         : "
            f"{result['cluster_members']}"
        )

        print("Investigation reasons:")

        for reason in result["reasons"]:
            print(f"  - {reason}")

        if result["pair_relationships"]:
            print("Direct relationships:")

            for linked_id, relationship in (
                result["pair_relationships"].items()
            ):
                print(
                    f"  {linked_id}: "
                    f"{relationship['signals']} "
                    f"({relationship['score']:.2%})"
                )

    print()
    print("=" * 75)
    print("IDENTITY GRAPH V2 TEST COMPLETED")
    print("=" * 75)
