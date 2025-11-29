# debug_fetch.py
import asyncio
import os
from agents import verifier   # adjust if your module path differs

async def run_debug(claim):
    print("Claim:", claim)
    # Call the same expansion + fetch logic used by verify_claim but print raw results
    variants = [claim]
    words = [w for w in __import__("re").split(r"\W+", claim) if len(w) > 3]
    if words:
        variants.append(" ".join(words[:6]))

    tasks = []
    tasks.append(asyncio.create_task(verifier.fetch_boom_live()))
    tasks.append(asyncio.create_task(verifier.fetch_afp()))
    for v in variants[:3]:
        tasks.append(asyncio.create_task(verifier.fetch_google_factchecks(v)))
        tasks.append(asyncio.create_task(verifier.fetch_pib_factcheck(v)))
        tasks.append(asyncio.create_task(verifier.fetch_factly(v)))
        tasks.append(asyncio.create_task(verifier.fetch_politifact(v)))
        tasks.append(asyncio.create_task(verifier.fetch_reuters(v)))
        tasks.append(asyncio.create_task(verifier.fetch_google_news(v)))
        tasks.append(asyncio.create_task(verifier.fetch_newsapi(v)))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    i = 0
    for res in results:
        i += 1
        print(f"\n--- Fetcher {i} returned ---")
        if isinstance(res, Exception):
            print("Exception:", res)
            continue
        if not res:
            print("[]")
            continue
        for r in res[:10]:
            print(f"[{r.get('type')}] {r.get('title')[:120]} — {r.get('url')}")
    # show embeddings for top title (if OPENAI active)
    if results and isinstance(results[0], list) and results[0]:
        sample = results[0][0].get("title") or results[0][0].get("snippet","")
        emb = await verifier.embed_text(sample)
        print("\nSample embedding vector length:", None if emb is None else len(emb))

if __name__ == "__main__":
    claim = "Meeting between India and Russia Defence Ministers to discuss S-500 acquisition"
    asyncio.run(run_debug(claim))
