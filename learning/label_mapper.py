from typing import Union


LABEL_MAP = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}


REVERSE_LABEL_MAP = {
    value.lower(): key
    for key, value in LABEL_MAP.items()
}


def label_to_name(label: int) -> str:
    label = int(label)

    if label not in LABEL_MAP:
        raise ValueError(
            f"Unknown TrustLoop label: {label}. "
            f"Expected one of {list(LABEL_MAP.keys())}."
        )

    return LABEL_MAP[label]


def name_to_label(name: str) -> int:
    if not isinstance(name, str):
        raise TypeError("Label name must be a string.")

    normalized = name.strip().lower()

    if normalized not in REVERSE_LABEL_MAP:
        raise ValueError(
            f"Unknown TrustLoop label name: {name}. "
            f"Expected one of {list(LABEL_MAP.values())}."
        )

    return REVERSE_LABEL_MAP[normalized]


def normalize_label(
    value: Union[int, str]
) -> int:

    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid labels.")

    if isinstance(value, int):
        return value if value in LABEL_MAP else (
            (_ for _ in ()).throw(
                ValueError(f"Invalid TrustLoop label: {value}")
            )
        )

    if isinstance(value, str):
        value = value.strip()

        if value.isdigit():
            numeric = int(value)

            if numeric in LABEL_MAP:
                return numeric

        return name_to_label(value)

    raise TypeError(
        "Label must be an integer or class name."
    )


def validate_label(label: int) -> bool:
    return int(label) in LABEL_MAP


if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP LABEL MAPPER")
    print("=" * 70)

    for label, name in LABEL_MAP.items():
        print(f"{label} -> {name}")

    print("\nValidation tests:")

    for value in [
        0,
        1,
        2,
        3,
        "Legitimate",
        "Policy Abuser",
        "Fraudulent Return",
        "Wardrobing",
    ]:
        normalized = normalize_label(value)

        print(
            f"{value!r} -> "
            f"{normalized} -> "
            f"{label_to_name(normalized)}"
        )

    print("\nLabel mapper ready.")
