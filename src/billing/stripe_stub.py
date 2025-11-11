"""Lightweight Stripe billing stub for local/testing environments."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class BillingRecord:
    customer_id: str
    email: str
    plan: str
    created_at: float


class StripeBillingStub:
    def __init__(self):
        self.records: Dict[str, BillingRecord] = {}

    def create_customer(self, email: str, plan: str = "basic") -> BillingRecord:
        identifier = f"cus_{int(time.time())}"
        record = BillingRecord(customer_id=identifier, email=email, plan=plan, created_at=time.time())
        self.records[identifier] = record
        return record

    def usage_summary(self, customer_id: str) -> dict:
        record = self.records.get(customer_id)
        if not record:
            raise KeyError(customer_id)
        return {
            "customer_id": record.customer_id,
            "plan": record.plan,
            "status": "stub",
        }


def get_billing_provider():
    mode = os.getenv("BILLING_MODE", "local")
    if mode == "stripe" and os.getenv("STRIPE_SECRET_KEY"):
        # Placeholder hook for real Stripe integration.
        return StripeBillingStub()
    return StripeBillingStub()
