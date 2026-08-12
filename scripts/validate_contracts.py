#!/usr/bin/env python3
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from packages.blueprint_schema.validator import BlueprintValidator


def main():
    print("==================================================")
    print("Video Blueprint Studio - Contract Validation Script")
    print("Canonical Schema: Draft 2020-12")
    print("==================================================")

    validator = BlueprintValidator()

    # 1. Validate Positive Fixture (example_blueprint.json)
    example_path = os.path.abspath("contracts/example_blueprint.json")
    print(f"\n[+] Validating positive fixture: {example_path}")
    if not os.path.exists(example_path):
        print(f"[-] ERROR: File not found: {example_path}")
        sys.exit(1)

    with open(example_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_valid, errors = validator.validate(data)
    if is_valid:
        print("[✔] Positive fixture PASS: example_blueprint.json is schema compliant.")
    else:
        print("[✘] Positive fixture FAIL:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    # 2. Validate Intentionally Invalid Fixtures
    fixtures_dir = os.path.abspath("tests/contract/fixtures")
    print(f"\n[+] Validating negative fixtures in: {fixtures_dir}")
    negative_passed = True

    for filename in sorted(os.listdir(fixtures_dir)):
        if filename.startswith("invalid_") and filename.endswith(".json"):
            filepath = os.path.join(fixtures_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                invalid_data = json.load(f)
            
            val_result, _ = validator.validate(invalid_data)
            if not val_result:
                print(f"[✔] Negative fixture correctly REJECTED: {filename}")
            else:
                print(f"[✘] Negative fixture UNEXPECTEDLY PASSED: {filename}")
                negative_passed = False

    if not negative_passed:
        print("[-] Contract validation failed on negative fixtures.")
        sys.exit(1)

    print("\n==================================================")
    print("ALL CONTRACT VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
