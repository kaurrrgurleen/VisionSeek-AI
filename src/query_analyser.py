import os
import json
from openai import OpenAI


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "openrouter/free"


# ==================================================
# OPENROUTER CLIENT
# ==================================================

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not configured."
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ==================================================
# QUERY ANALYZER
# ==================================================

def analyze_query(user_query):

    system_prompt = """
You are the query understanding module of VisionSeek AI,
a semantic video search system.

Analyze the user's video-search query and convert it
into structured search intent.

Return ONLY valid JSON with exactly these fields:

{
    "visual_query": "...",
    "objects": [],
    "actions": [],
    "scene": "...",
    "temporal": true
}

Rules:

1. visual_query:
   Rewrite the query into a concise visual description
   that a vision-language model such as CLIP can understand.

2. objects:
   List the important visible objects.

3. actions:
   List actions or events mentioned by the user.

4. scene:
   Describe the environment or setting.

5. temporal:
   Set true if the query involves an action, movement,
   sequence, before/after relationship, or event over time.
   Otherwise set false.

Do not invent objects or actions that are not implied
by the user's query.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=0,
        response_format={
        "type": "json_object"
        }
    )

    content = response.choices[0].message.content
    # Remove accidental markdown fences
    content = content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:

        result = json.loads(content)

        return result

    except json.JSONDecodeError:

        print("\nWarning: LLM returned an unexpected response.")
        print("Attempting fallback query analysis...")

        # --------------------------------------------------
        # FALLBACK ANALYSIS
        # --------------------------------------------------

        fallback_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
    Convert the user's video search query into JSON.

    Return ONLY JSON.

    Use exactly this structure:

    {
        "visual_query": "short visual description",
        "objects": [],
        "actions": [],
        "scene": "scene description",
        "temporal": false
    }

    Do not include explanations.
    Do not include Markdown.
    """
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            temperature=0
        )

        fallback_content = (
            fallback_response
            .choices[0]
            .message
            .content
            .strip()
        )

        # Remove Markdown fences if present
        if fallback_content.startswith("```"):

            fallback_content = (
                fallback_content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        try:

            result = json.loads(
                fallback_content
            )

            return result

        except json.JSONDecodeError:

            print(
                "\nFallback analysis also failed."
            )

            # --------------------------------------------------
            # LOCAL FALLBACK
            # --------------------------------------------------

            return {
                "visual_query": user_query,
                "objects": [],
                "actions": [],
                "scene": "unknown",
                "temporal": False
            }

# ==================================================
# TEST
#    ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VISIONSEEK — LLM QUERY ANALYZER")
    print("=" * 60)

    while True:

        query = input(
            "\nEnter a video search query "
            "(or type 'exit'): "
        )

        if query.lower() == "exit":
            break

        if not query.strip():
            print("Please enter a query.")
            continue

        try:

            result = analyze_query(query)

            print("\nStructured Search Intent")
            print("-" * 60)

            print(
                json.dumps(
                    result,
                    indent=4
                )
            )

        except Exception as e:

            print("\nError:")
            print(e)
