from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CustomerNode:
    customer_id: str
    address: str = ""
    phone: str = ""
    device_id: str = ""
    payment_id: str = ""


class IdentityFraudGraph:

    def __init__(self):
        self.customers = {}
        self.indexes = {
            "address": defaultdict(set),
            "phone": defaultdict(set),
            "device_id": defaultdict(set),
            "payment_id": defaultdict(set),
        }

    def add_customer(
        self,
        customer_id,
        address="",
        phone="",
        device_id="",
        payment_id="",
    ):
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

    def _linked_customers(self, customer_id):
        if customer_id not in self.customers:
            return set()

        node = self.customers[customer_id]
        linked = set()

        values = {
            "address": node.address,
            "phone": node.phone,
            "device_id": node.device_id,
            "payment_id": node.payment_id,
        }

        for field_name, value in values.items():
            if value:
                linked.update(self.indexes[field_name].get(value, set()))

        linked.discard(customer_id)

        return linked

    def analyze_customer(self, customer_id):

        if customer_id not in self.customers:
            raise ValueError(
                f"Customer '{customer_id}' does not exist."
            )

        node = self.customers[customer_id]

        linked = self._linked_customers(customer_id)

        shared_address_count = max(
            0,
            len(self.indexes["address"].get(node.address, set())) - 1
        ) if node.address else 0

        shared_phone_count = max(
            0,
            len(self.indexes["phone"].get(node.phone, set())) - 1
        ) if node.phone else 0

        shared_device_count = max(
            0,
            len(self.indexes["device_id"].get(node.device_id, set())) - 1
        ) if node.device_id else 0

        shared_payment_count = max(
            0,
            len(self.indexes["payment_id"].get(node.payment_id, set())) - 1
        ) if node.payment_id else 0

        # Conservative graph risk.
        # Shared identifiers create investigation signals,
        # not automatic fraud conclusions.
        score = (
            min(shared_address_count * 0.08, 0.25)
            + min(shared_phone_count * 0.15, 0.30)
            + min(shared_device_count * 0.20, 0.35)
            + min(shared_payment_count * 0.25, 0.40)
        )

        score = min(score, 1.0)

        if score >= 0.60:
            risk_level = "HIGH"
        elif score >= 0.30:
            risk_level = "MEDIUM"
        elif score > 0:
            risk_level = "LOW"
        else:
            risk_level = "NONE"

        return {
            "customer_id": customer_id,
            "shared_address_count": shared_address_count,
            "shared_phone_count": shared_phone_count,
            "shared_device_count": shared_device_count,
            "shared_payment_count": shared_payment_count,
            "linked_customer_count": len(linked),
            "linked_customers": sorted(linked),
            "identity_link_score": round(score, 4),
            "risk_level": risk_level,
        }


if __name__ == "__main__":

    graph = IdentityFraudGraph()

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

    print("=" * 70)
    print("TRUSTLOOP — IDENTITY & ADDRESS FRAUD GRAPH TEST")
    print("=" * 70)

    for customer_id in ["C001", "C002", "C003", "C004"]:

        result = graph.analyze_customer(customer_id)

        print()
        print(f"CUSTOMER: {customer_id}")
        print("-" * 70)

        print(
            f"Shared address     : "
            f"{result['shared_address_count']}"
        )

        print(
            f"Shared phone       : "
            f"{result['shared_phone_count']}"
        )

        print(
            f"Shared device      : "
            f"{result['shared_device_count']}"
        )

        print(
            f"Shared payment     : "
            f"{result['shared_payment_count']}"
        )

        print(
            f"Linked customers   : "
            f"{result['linked_customer_count']}"
        )

        print(
            f"Identity link score: "
            f"{result['identity_link_score']:.2%}"
        )

        print(
            f"Risk level         : "
            f"{result['risk_level']}"
        )

        print(
            f"Linked IDs         : "
            f"{result['linked_customers']}"
        )

    print()
    print("=" * 70)
    print("IDENTITY GRAPH TEST COMPLETED")
    print("=" * 70)
