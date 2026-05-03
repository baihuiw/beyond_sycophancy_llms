#!/usr/bin/env python
"""Test script to verify API keys and logprob support.

Usage:
    python test_api_setup.py --provider azure
    python test_api_setup.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct:free
"""

import argparse
import asyncio
import os
import sys

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    print("[ERROR] OpenAI library not installed. Run: pip install openai")
    sys.exit(1)

# API key placeholders (same as in exp1_inference_logits.py)
AZURE_OPENAI_API_KEY = ""
OPENROUTER_API_KEY = ""


def get_env_or_placeholder(env_name: str, placeholder: str) -> str:
    """Get value from environment variable or use placeholder."""
    v = os.getenv(env_name, "").strip()
    if v:
        return v
    return (placeholder or "").strip()


async def test_azure(endpoint: str, api_version: str, deployment: str):
    """Test Azure OpenAI API with logprobs."""
    print("\n" + "=" * 60)
    print("TESTING AZURE OPENAI")
    print("=" * 60)

    api_key = get_env_or_placeholder("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY)

    if not api_key:
        print("[ERROR] No API key found. Set AZURE_OPENAI_API_KEY or fill placeholder.")
        return False

    print(f"Endpoint: {endpoint}")
    print(f"API Version: {api_version}")
    print(f"Deployment: {deployment}")
    print(f"API Key: {'*' * 20}{api_key[-4:] if len(api_key) > 4 else '****'}")

    try:
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

        print("\n[TEST 1] Basic API call...")
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'hello' in one word."}
            ],
            max_tokens=5,
        )

        print(f"✓ API call successful")
        print(f"  Response: {response.choices[0].message.content}")

        print("\n[TEST 2] Logprobs support...")
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Answer according to your best judgment. Answer in English."},
                {"role": "user",
                 "content": "On a scale of 1 to 7, how much do you agree? Answer: 4\n\nRespond with exactly one character: one of 1,2,3,4,5,6,7. No other text."}
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            temperature=0.0,
        )

        if response.choices[0].logprobs:
            content_logprobs = response.choices[0].logprobs.content
            if content_logprobs and len(content_logprobs) > 0:
                top_logprobs = content_logprobs[0].top_logprobs

                print(f"✓ Logprobs supported! Got {len(top_logprobs)} tokens")

                # Check for Likert tokens
                likert_tokens = ["1", "2", "3", "4", "5", "6", "7"]
                found = {item.token for item in top_logprobs if item.token in likert_tokens}

                print(f"  Likert tokens found: {sorted(found)}")

                if len(found) == 7:
                    print("  ✓ ALL 7 Likert tokens present!")
                else:
                    missing = set(likert_tokens) - found
                    print(f"  ⚠ Missing tokens: {sorted(missing)}")
                    print(f"  → Try increasing top_logprobs in config")

                # Show sample logprobs
                print("\n  Sample logprobs:")
                for item in top_logprobs[:5]:
                    print(f"    {item.token:5s} → {item.logprob:.4f}")

                return True

        print("✗ Logprobs not returned")
        print("  This deployment may not support logprobs.")
        return False

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {str(e)}")
        return False


async def test_openrouter(model: str):
    """Test OpenRouter API with logprobs."""
    print("\n" + "=" * 60)
    print("TESTING OPENROUTER")
    print("=" * 60)

    api_key = get_env_or_placeholder("OPENROUTER_API_KEY", OPENROUTER_API_KEY)

    if not api_key:
        print("[ERROR] No API key found. Set OPENROUTER_API_KEY or fill placeholder.")
        return False

    print(f"Model: {model}")
    print(f"API Key: {'*' * 20}{api_key[-4:] if len(api_key) > 4 else '****'}")

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        print("\n[TEST 1] Basic API call...")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'hello' in one word."}
            ],
            max_tokens=5,
        )

        print(f"✓ API call successful")
        print(f"  Response: {response.choices[0].message.content}")

        print("\n[TEST 2] Logprobs support...")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer according to your best judgment. Answer in English."},
                {"role": "user",
                 "content": "On a scale of 1 to 7, how much do you agree? Answer: 4\n\nRespond with exactly one character: one of 1,2,3,4,5,6,7. No other text."}
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            temperature=0.0,
        )

        if response.choices[0].logprobs:
            content_logprobs = response.choices[0].logprobs.content
            if content_logprobs and len(content_logprobs) > 0:
                top_logprobs = content_logprobs[0].top_logprobs

                print(f"✓ Logprobs supported! Got {len(top_logprobs)} tokens")

                # Check for Likert tokens
                likert_tokens = ["1", "2", "3", "4", "5", "6", "7"]
                found = {item.token for item in top_logprobs if item.token in likert_tokens}

                print(f"  Likert tokens found: {sorted(found)}")

                if len(found) == 7:
                    print("  ✓ ALL 7 Likert tokens present!")
                else:
                    missing = set(likert_tokens) - found
                    print(f"  ⚠ Missing tokens: {sorted(missing)}")
                    print(f"  → Try increasing top_logprobs in config")

                # Show sample logprobs
                print("\n  Sample logprobs:")
                for item in top_logprobs[:5]:
                    print(f"    {item.token:5s} → {item.logprob:.4f}")

                return True

        print("✗ Logprobs not returned")
        print("  This model may not support logprobs on OpenRouter.")
        print("  → Use stochastic sampling (exp1_inference_hydra.py) instead.")
        return False

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {str(e)}")

        if "401" in str(e) or "authentication" in str(e).lower():
            print("\n  → Check your API key")
        elif "404" in str(e) or "not found" in str(e).lower():
            print("\n  → Model not found. Check model slug at https://openrouter.ai/models")
        elif "429" in str(e) or "rate limit" in str(e).lower():
            print("\n  → Rate limited. Wait and try again.")

        return False


def main():
    parser = argparse.ArgumentParser(description="Test API setup for logit extraction")
    parser.add_argument(
        "--provider",
        choices=["azure", "openrouter"],
        required=True,
        help="API provider to test"
    )
    parser.add_argument(
        "--endpoint",
        default="",
        help="Azure endpoint (for Azure only)"
    )
    parser.add_argument(
        "--api-version",
        default="2024-12-01-preview",
        help="Azure API version (for Azure only)"
    )
    parser.add_argument(
        "--deployment",
        default="gpt-4o",
        help="Azure deployment name (for Azure only)"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3.3-70b-instruct:free",
        help="OpenRouter model slug (for OpenRouter only)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("API SETUP TEST")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Test basic API connectivity")
    print("2. Verify logprobs support")
    print("3. Check if all 7 Likert tokens are captured")
    print("")

    if args.provider == "azure":
        success = asyncio.run(test_azure(
            args.endpoint,
            args.api_version,
            args.deployment
        ))
    else:
        success = asyncio.run(test_openrouter(args.model))

    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nYou're ready to run experiments!")
        print("\nNext steps:")
        if args.provider == "azure":
            print("  ./run_logit_experiment_api.sh exp1_logits_azure_gpt4o baseline")
        else:
            model_key = args.model.replace("/", "_").replace(":", "_")
            print(f"  ./run_logit_experiment_api.sh exp1_logits_openrouter_{model_key} baseline")
    else:
        print("✗ TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before running experiments.")
        print("\nCommon fixes:")
        print("  - Set API key: export AZURE_OPENAI_API_KEY='...'")
        print("  - Check deployment name (Azure)")
        print("  - Check model slug (OpenRouter)")
        print("  - Verify model supports logprobs")
    print("")


if __name__ == "__main__":
    main()