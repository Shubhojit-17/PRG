"""Seed Phoenix with demo prompts and datasets for local testing."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from config.settings import settings

PHOENIX_API_PREFIX = "/api"
PHOENIX_PROJECTS_ENDPOINT = f"{PHOENIX_API_PREFIX}/projects"
PHOENIX_PROMPTS_ENDPOINT = f"{PHOENIX_API_PREFIX}/prompts"
PHOENIX_DATASETS_ENDPOINT = f"{PHOENIX_API_PREFIX}/datasets"
REQUEST_TIMEOUT_SECONDS = 10

PROMPT_NAME = "customer-support-assistant"
BASELINE_TAG = "v1.0"
CANDIDATE_TAG = "v1.1"
BASELINE_LABEL = "stable"

logger = structlog.get_logger()

EXAMPLES = [
    {
        "input": "My order has not arrived. What should I do?",
        "expected_output": "I am sorry your order is delayed. Please check your tracking link for the latest status. If it is past the estimated delivery date, share your order number and I will open a delivery investigation right away.",
    },
    {
        "input": "I was charged twice for the same purchase. Can you fix it?",
        "expected_output": "Thanks for letting us know. Please send the order number and a screenshot of the charges. We will verify the duplicate and issue a refund within 3 to 5 business days.",
    },
    {
        "input": "The item I received is damaged. How do I get a replacement?",
        "expected_output": "I am sorry about that. Please share photos of the damage and your order number. We will arrange a replacement or refund, whichever you prefer.",
    },
    {
        "input": "I need to change my shipping address.",
        "expected_output": "I can help with that. Please provide your order number and the correct shipping address. If the order has not shipped, we will update it immediately.",
    },
    {
        "input": "Can I return a product I bought last week?",
        "expected_output": "Yes, returns are accepted within 30 days of delivery. Please share your order number and we will send return instructions and a prepaid label.",
    },
    {
        "input": "I canceled my order but still received it. What now?",
        "expected_output": "I am sorry for the mix-up. Please keep the packaging and let us know if you would like a return pickup or a prepaid label. We will make it right.",
    },
    {
        "input": "The tracking says delivered, but I do not have the package.",
        "expected_output": "Thanks for checking. Please confirm your address and look for a delivery photo or a notice. If it is still missing, we will open a claim with the carrier and send a replacement if needed.",
    },
    {
        "input": "How long does standard shipping take?",
        "expected_output": "Standard shipping typically takes 3 to 5 business days after the order ships. You can see the exact estimate at checkout and in your tracking email.",
    },
    {
        "input": "My promo code did not apply. Can you help?",
        "expected_output": "Absolutely. Please share the promo code and your cart details. I will check eligibility and apply a discount if it should have worked.",
    },
    {
        "input": "I need an invoice for my purchase.",
        "expected_output": "No problem. Please provide your order number and billing email. I will send a PDF invoice to you shortly.",
    },
]

BASELINE_PROMPT = (
    "You are a customer support assistant for an e-commerce company. Provide clear, "
    "polite, and professional help. Always acknowledge the customer issue, ask for "
    "the minimum details needed to resolve it, and explain next steps. If a request "
    "is outside policy, offer the closest alternative. Keep responses concise, "
    "actionable, and friendly. Never request sensitive information like passwords or "
    "full credit card numbers. When appropriate, confirm timelines for refunds, "
    "shipping, or escalations. Maintain a calm, respectful tone and avoid slang. "
    "Use short paragraphs and bullet points when helpful so the customer can scan "
    "the response easily."
)

CANDIDATE_PROMPT = (
    "You are a support helper. Try to be friendly and casual. Respond however seems "
    "right and keep it short. You can be flexible with details and do not worry too "
    "much about formal steps. The goal is to make the customer feel okay, even if "
    "the response is not super specific."
)


def build_headers() -> dict[str, str]:
    """Build HTTP headers for Phoenix API requests.

    Returns:
        dict[str, str]: HTTP headers.
    """

    headers = {"Content-Type": "application/json"}
    if settings.phoenix.api_key:
        headers["Authorization"] = f"Bearer {settings.phoenix.api_key}"
    return headers


async def get_or_create_project(client: httpx.AsyncClient) -> str:
    """Get or create a Phoenix project by name.

    Args:
        client: Async HTTP client.

    Returns:
        str: Project identifier.
    """

    response = await client.get(PHOENIX_PROJECTS_ENDPOINT)
    response.raise_for_status()
    projects = response.json()
    for project in projects:
        if project.get("name") == settings.phoenix.project_name:
            return str(project.get("id"))

    create_response = await client.post(
        PHOENIX_PROJECTS_ENDPOINT,
        json={"name": settings.phoenix.project_name},
    )
    create_response.raise_for_status()
    return str(create_response.json().get("id"))


async def create_prompt_and_versions(client: httpx.AsyncClient, project_id: str) -> str:
    """Create a prompt and its versions in Phoenix.

    Args:
        client: Async HTTP client.
        project_id: Project identifier.

    Returns:
        str: Prompt identifier.
    """

    prompt_response = await client.post(
        PHOENIX_PROMPTS_ENDPOINT,
        json={"name": PROMPT_NAME, "project_id": project_id},
    )
    prompt_response.raise_for_status()
    prompt_id = str(prompt_response.json().get("id"))

    baseline_response = await client.post(
        f"{PHOENIX_PROMPTS_ENDPOINT}/{prompt_id}/versions",
        json={
            "prompt_text": BASELINE_PROMPT,
            "version_tag": BASELINE_TAG,
            "labels": [BASELINE_LABEL],
        },
    )
    baseline_response.raise_for_status()

    candidate_response = await client.post(
        f"{PHOENIX_PROMPTS_ENDPOINT}/{prompt_id}/versions",
        json={
            "prompt_text": CANDIDATE_PROMPT,
            "version_tag": CANDIDATE_TAG,
            "labels": [],
        },
    )
    candidate_response.raise_for_status()

    return prompt_id


async def create_dataset(client: httpx.AsyncClient, project_id: str) -> None:
    """Create a demo evaluation dataset.

    Args:
        client: Async HTTP client.
        project_id: Project identifier.
    """

    dataset_payload: dict[str, Any] = {
        "name": PROMPT_NAME,
        "project_id": project_id,
        "examples": EXAMPLES,
    }
    response = await client.post(PHOENIX_DATASETS_ENDPOINT, json=dataset_payload)
    response.raise_for_status()


async def seed() -> None:
    """Seed Phoenix with demo data."""

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    logger.info(
        "seeding_phoenix",
        host=settings.phoenix.host,
        project=settings.phoenix.project_name,
    )
    async with httpx.AsyncClient(
        base_url=settings.phoenix.host, timeout=timeout, headers=build_headers()
    ) as client:
        project_id = await get_or_create_project(client)
        await create_prompt_and_versions(client, project_id)
        await create_dataset(client, project_id)

    print(
        f"Seeded demo data in Phoenix at {settings.phoenix.host}/projects/{project_id}"
    )


def main() -> None:
    """Entry point for the seed script."""

    asyncio.run(seed())


if __name__ == "__main__":
    main()
