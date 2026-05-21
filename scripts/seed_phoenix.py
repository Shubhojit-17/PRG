import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()  # load .env FIRST

api_key = os.environ.get("PHOENIX_API_KEY", "")
phoenix_host = os.environ.get("PHOENIX_HOST", "https://app.phoenix.arize.com")

# Verify before proceeding
print(f"Host: {phoenix_host}")
print(
    f"API key present: {bool(api_key)}, starts with: {api_key[:8] if api_key else 'MISSING'}"
)

if not api_key:
    raise ValueError("PHOENIX_API_KEY is not set in .env")

import phoenix as px
from phoenix.client import Client
from phoenix.client.types import PromptVersion  # inspect exact import path first


def seed() -> None:
    client = Client(
        base_url=phoenix_host,
        api_key=api_key,
    )

    # -- 1. Create baseline prompt (v1.0 - stable, professional) --
    baseline_version = client.prompts.create(
        name="customer-support-assistant",
        prompt_description="Customer support chatbot - baseline stable version",
        version=PromptVersion(
            [
                {
                    "role": "system",
                    "content": "You are a customer support assistant for an ecommerce company. Your goal is to resolve customer issues clearly, politely, and efficiently. Always acknowledge the problem, show empathy, and confirm the next step. Ask only for the minimum details needed, such as order number, email used at checkout, and shipping address when relevant. Provide concise guidance for delays, damaged items, returns, refunds, replacements, cancellations, and billing questions. State typical timelines for shipping, refunds, or investigations. If a request is outside policy, explain the policy briefly and offer the closest alternative or escalation path. Never request passwords, full card numbers, or other sensitive credentials. Keep responses short, professional, and easy to scan with short paragraphs or bullets. When information is missing, ask a single clear follow up question. Do not guess order status or promise outcomes you cannot verify.",
                },
                {"role": "user", "content": "{{question}}"},
            ],
            model_name="gpt-4o-mini",
            description="Baseline prompt version v1.0",
            model_provider="OPENAI",
            template_format="MUSTACHE",
        ),
    )
    print(f"Created baseline prompt version: {baseline_version}")

    # -- 2. Tag baseline as stable --
    client.prompts.tags.create(
        prompt_version_id=baseline_version.id,
        name="stable",
        description="Stable baseline version v1.0",
    )

    # -- 3. Create degraded prompt (v1.1 - intentionally worse) --
    candidate_version = client.prompts.create(
        name="customer-support-assistant",
        prompt_description="Degraded version - vague and overly casual",
        version=PromptVersion(
            [
                {
                    "role": "system",
                    "content": "You are a support helper. Keep things casual and upbeat. Answer in a friendly way without getting into too many details. It is fine to be general and reassure the customer. You do not need to ask for specific information unless it feels important. Skip strict policy talk and avoid long explanations. Keep replies short, informal, and flexible, and try to make the customer feel okay even if the answer is not precise.",
                },
                {"role": "user", "content": "{{question}}"},
            ],
            model_name="gpt-4o-mini",
            description="Candidate prompt version v1.1",
            model_provider="OPENAI",
            template_format="MUSTACHE",
        ),
    )
    print(f"Created candidate prompt version: {candidate_version}")

    # -- 4. Create evaluation dataset --
    examples = [
        {
            "input": {"question": "My order hasn't arrived, what do I do?"},
            "output": {
                "answer": "I'm sorry to hear that. Please check your tracking number for the latest status. If it is past the estimated delivery date, share your order number and I will open a delivery investigation."
            },
        },
        {
            "input": {"question": "I was charged twice for the same purchase. Can you fix it?"},
            "output": {
                "answer": "Thanks for letting us know. Please send the order number and a screenshot of the duplicate charges. We will verify and issue a refund within 3 to 5 business days."
            },
        },
        {
            "input": {"question": "The item I received is damaged. How do I get a replacement?"},
            "output": {
                "answer": "I'm sorry about that. Please share photos of the damage and your order number. We can ship a replacement or process a refund, whichever you prefer."
            },
        },
        {
            "input": {"question": "I need to change my shipping address."},
            "output": {
                "answer": "I can help. Please provide your order number and the correct address. If the order has not shipped yet, we will update it immediately."
            },
        },
        {
            "input": {"question": "Can I return a product I bought last week?"},
            "output": {
                "answer": "Yes, returns are accepted within 30 days of delivery. Share your order number and we will send return instructions and a prepaid label."
            },
        },
        {
            "input": {"question": "I canceled my order but it still arrived. What now?"},
            "output": {
                "answer": "Sorry for the mix up. Please let us know if you prefer a return pickup or a prepaid label. We will make it right."
            },
        },
        {
            "input": {"question": "Tracking says delivered, but I do not have the package."},
            "output": {
                "answer": "Please confirm your address and check for a delivery photo or notice. If it is still missing, we will open a carrier claim and arrange a replacement if needed."
            },
        },
        {
            "input": {"question": "How long does standard shipping take?"},
            "output": {
                "answer": "Standard shipping usually takes 3 to 5 business days after the order ships. You can see the estimate at checkout and in your tracking email."
            },
        },
        {
            "input": {"question": "My promo code did not apply. Can you help?"},
            "output": {
                "answer": "Absolutely. Share the promo code and your cart details so I can check eligibility and apply a discount if it should have worked."
            },
        },
        {
            "input": {"question": "I need an invoice for my purchase."},
            "output": {
                "answer": "No problem. Please provide your order number and billing email and I will send a PDF invoice."
            },
        },
    ]
    dataset = client.datasets.create_dataset(
        name="customer-support-assistant",
        examples=examples,
        input_keys=["question"],
        output_keys=["answer"],
        dataset_description="Eval dataset for customer support prompt regression testing",
    )
    print(f"Created dataset: {dataset}")
    print("\nSuccess! View your data at: https://app.phoenix.arize.com")


if __name__ == "__main__":
    seed()
