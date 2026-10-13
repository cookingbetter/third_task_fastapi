from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "monthly_fee": 9.99,
                    "usage_hours": 27.92,
                    "support_requests": 1,
                    "account_age_months": 14,
                    "failed_payments": 1,
                    "region": "america",
                    "device_type": "desktop",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                }
            ]
        }
    }


class DatasetRowChurn(FeatureVectorChurn):
    churn: int
