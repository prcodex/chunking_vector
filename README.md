# M3xA + M3xBr — Bedrock RAG Architecture v5

> **Status**: Design spec, not yet implemented. This document defines
> the target architecture for migrating a production multilingual
> financial-intelligence RAG system from a single-vector-per-document
> approach to a Bedrock-native, multi-modal, multi-layer chunking
> and embedding stack.
>
> Phase rollout tracked in [Issue #1](../../issues/1).

## Table of contents

- [TL;DR](#tldr)
- [1 · Context](#1--context)
- [2 · Embedding stack — exact model picks](#2--embedding-stack--exact-model-picks)
- [3 · Per-source chunking matrix](#3--per-source-chunking-matrix)
- [4 · Universal layer: Anthropic Contextual Retrieval](#4--universal-layer-anthropic-contextual-retrieval)
- [5 · Special case: Bank PDF triple-layer pattern](#5--special-case-bank-pdf-triple-layer-pattern)
- [6 · Special case: Podcast dual text+audio embedding](#6--special-case-podcast-dual-textaudio-embedding)
- [7 · Storage architecture](#7--storage-architecture)
- [8 · Hybrid retrieval flow with this stack](#8--hybrid-retrieval-flow-with-this-stack)
- [9 · M3xA vs M3xBr — what differs](#9--m3xa-vs-m3xbr--what-differs)
- [10 · Advanced techniques — what's in v5, what's deferred](#10--advanced-techniques--whats-in-v5-whats-deferred)
- [11 · Cost analysis](#11--cost-analysis)
- [12 · Phased rollout](#12--phased-rollout)
- [13 · Open questions](#13--open-questions)
- [15 · Reference](#15--reference)

## TL;DR

Move embedding + reranking off Voyage (not on Bedrock) onto a Bedrock-native stack tuned for multilingual financial research:

| Component | Bedrock model | Dim | Role |
|---|---|---|---|
| Text embedder | `cohere.embed-v4` | 1536 (Matryoshka) | All text content, EN + PT unified space |
| Audio embedder | `amazon.nova-multimodal-embed-v1` | 3072 (Matryoshka) | Podcast audio, native 30-sec segments |
| PDF page embedder | `cohere.embed-v4` (image mode) | 1536 | Bank PDF pages as images, preserves charts/tables |
| Reranker | `cohere.rerank-v3-5` | — | Finance-tuned, PT-supported, top-150 → top-30 |
| Context generator | `claude-haiku-4-5-20251001` | — | Anthropic contextual-retrieval prefix per chunk |

Chunking shifts from "1 vector per doc" to a **per-source-class strategy** with three universal layers:
1. Anthropic Contextual Retrieval prefix (Haiku-generated) on every chunk before embed
2. Hierarchical parent-child for long docs (parent 2000 / child 400-512 / 100 child overlap)
3. Dual-modality coverage for high-value sources (PDFs get triple-layer; podcasts get text + audio)

Same embedding model serves M3xA and M3xBr — Cohere v4's multilingual unified space removes the need for separate PT and EN indexes. Only BM25 stays language-split (existing en_stem / pt_stem Tantivy plan).

Phased rollout: 7 phases, ~10-12 weeks. Phases 1-4 are the 80/20 (new embedder, hierarchical, contextual retrieval, rerank). Phases 5-7 add multimodal PDFs, dual podcast, optional multi-scale.

---

## 1 · Context

### What's being replaced

Current state (per the internal ingestion pipeline spec and current architecture documentation):
- Voyage-3-large @ 2048d as the sole embedder
- One vector per document for virtually all content types (explicitly: <strong>one vector per podcast episode</strong>, no chunking — confirmed by the current architecture documentation)
- Hybrid retrieval planned but not yet built (per the predecessor hybrid retrieval design)
- Reranking not yet wired

### Why Bedrock-native now

1. **Voyage isn't on Bedrock**, and the corporate AWS deployment pattern explored earlier this year benefits from native Bedrock APIs (less infra to manage, IAM-clean, batch inference, no separate vendor key).
2. **Cohere Embed v4** (April 2025 release, Bedrock GA later 2025) is the first multimodal+multilingual+finance-tuned embedder in a single model.
3. **Amazon Nova Multimodal Embeddings** (GA October 2025) adds native audio embedding — directly addresses the known podcast retrieval black hole.
4. **Cohere Rerank 3.5** (Bedrock GA December 2024) finance-tuned, multilingual.

### Why quality, not cost, drives this

Cost is not a constraint for this work. Estimated steady-state cost of this stack is ~$90/month at current volumes; one-time backfill ~$400. See §11.

### Anonymization convention for this file

This file uses anonymized source labels (Bank 1–11, Bank A–F, Wire 1–7, Pub 1–N, Indie 1–N) so it can be lifted directly into a public GitHub repo without exposing the source mix. The real source name map lives only in an internal source-mapping document. Any future edit to this file must preserve the anonymization.

---

## 2 · Embedding stack — exact model picks

### 2.1 Primary text embedder: Cohere Embed v4

| Spec | Value |
|---|---|
| Bedrock model ID | `cohere.embed-v4` |
| Default dimensions | 1536 (Matryoshka: 256 / 512 / 1024 / 1536) |
| Max context | 128K tokens |
| Languages | 100+ including Portuguese, Arabic |
| Modalities | Text + images (interleaved supported) |
| MTEB English score | ~65.2-65.8 (SOTA tier) |
| Bedrock pricing | $0.12 / 1M input tokens |
| Bedrock regions | us-east-1, eu-west-1, ap-northeast-1 (+ cross-region inference) |

**Why this and not Titan v2:**
- Multilingual: Titan v2 is English-strong, weak elsewhere. M3xBr is Portuguese-first.
- Financial domain tuning: Cohere v4 is explicitly fine-tuned for finance, healthcare, manufacturing. Titan has no domain tuning.
- Multimodal: Bank PDFs are chart/table-heavy. Cohere v4 handles images natively.
- 128K context: most long-form docs can be embedded whole if needed for a doc-level vector.

**Dimension choice: 1536 (full).** Storage at this corpus size (~500K-1M chunks after rollout) is ~3-6 GB — trivial. Matryoshka truncation to 1024 or 512 stays available as a future optimization if ANN latency becomes a bottleneck.

### 2.2 Audio embedder: Amazon Nova Multimodal Embeddings

| Spec | Value |
|---|---|
| Bedrock model ID | `amazon.nova-multimodal-embed-v1` |
| Default dimensions | 3072 (Matryoshka: 256 / 384 / 1024 / 3072) |
| Audio/video segment | up to 30 sec, auto-segmentation for longer files |
| Text context | up to 8K tokens |
| Languages | 200+ |
| Bedrock region (GA) | us-east-1 |

**Why dual-embed podcasts:** Transcription-only loses non-verbal information (speaker tone, pace shifts, market reactions in real time on financial podcasts). Nova embeds raw audio in a unified vector space, enabling semantic audio queries the text path cannot serve.

**Dimension choice: 3072 (full)** for maximum semantic resolution. Audio corpus is small relative to text (~10 podcasts/day × ~2hr).

### 2.3 PDF page-image embedder: Cohere Embed v4 (image mode)

Same model as 2.1, called via the image mode of the Bedrock API. Each PDF page rendered to image (2MM pixels max), passed to Cohere v4. Output sits in the same 1536d space as text — cross-modal retrieval just works.

This is the **ColPali / ViDoRe pattern**: skip OCR entirely for visual content. Catches equity curves, rate scatter plots, regional GDP heatmaps, formatted tables that text extraction destroys.

### 2.4 Reranker: Cohere Rerank 3.5

| Spec | Value |
|---|---|
| Bedrock model ID | `cohere.rerank-v3-5` |
| Context | 4096 tokens |
| Languages | 100+; SOTA on 10 business languages incl. Portuguese |
| Finance benchmark | +23.4% NDCG@10 vs Hybrid Search, +30.8% vs BM25 |

Drop-in replacement for the Voyage rerank-2.5 line previously planned for the rerank stage of the hybrid pipeline. Same instruction-following pattern — rewriter emits per-query rerank instructions derived from soul rules.

### 2.5 Context generator: Claude Haiku 4.5

| Spec | Value |
|---|---|
| Bedrock model ID | `claude-haiku-4-5-20251001` (or successor) |
| Role | Generate 50-100 token contextual prefix per chunk at ingestion time |

Already in the stack as Actor 1 classifier. Extends to new role at ingestion.

---

## 3 · Per-source chunking matrix

### 3.1 M3xA (macro/global domain)

| Source class | Anonymized sources | Avg tokens | Chunking strategy | Embedder | Vector count per doc |
|---|---|---|---|---|---|
| Macro tweets | ~130 accounts (wires, central bankers, analysts, geo voices) | 30-80 | 1 vector + contextual prefix | Cohere v4 1536d | 1 |
| Wire articles | Wire 1-7 (terminal wires, financial dailies, market wires, global business, regional broadsheet, defense wire) | 400-2000 | If <1500 tok → 1 vec; else hierarchical (parent=full, child 512 / overlap 100) | Cohere v4 1536d | 1 or 1+N |
| Long-form journalism | Pub 1-8 (policy magazines, defense pubs, regional broadsheets, NYT verticals, dashboards) | 3000-8000 | Hierarchical: parent 2000 / child 400 / parent overlap 200 / child overlap 80, section-aware (markdown headers) | Cohere v4 1536d | 1 doc + N children |
| Email research notes | Bank 1 desk, Bank 5 alt, Indie 1-2, EM-desk 1-2, Sector desk 1-2, Substack ×16, Bank 8 desk, Indie blogs ×3 | 1500-6000 | Section-aware split on headings → 400-600 tok chunks / 80 overlap | Cohere v4 1536d | N (typically 3-15) |
| Web scrapers | Wire 8-9, alt-finance, market wire 2, Indie 3-12, central bank scrapers, polling tracker | 800-3000 | Same as wires | Cohere v4 1536d | 1 or 1+N |
| Drive PDFs | Bank 1-11 (sell-side and buy-side research desks) | 5K-25K | **Triple-layer**: (A) doc-level vec, (B) hierarchical text (parent 2000 / child 512 / overlap 100), (C) page-image vec per page | Cohere v4 1536d both text + image modes | 1 + N children + P pages |
| Podcast transcripts | 10 RSS audio (Bank pods + wire pods + macro pod) + 9 YouTube geo channels | 5K-25K | Semantic + timestamp-aligned: ~600 tok (~5min audio) chunks, 100 tok (~1min) overlap, speaker-turn aware | Cohere v4 1536d (text) + Nova 3072d (audio parallel) | M text chunks + K audio segments |
| Geo OSINT structured | Hormuz monitor, Iran proxies, Iran intel scrapers | mixed | Short events: 1 vec; long reports: article-style | Cohere v4 1536d | 1 or N |
| Calendar / markets / Polymarket | structured | structured | No chunking — DB rows, joined by time at query time | n/a | n/a |

### 3.2 M3xBr (Brazil domain)

| Source class | Anonymized sources | Avg tokens | Chunking strategy | Embedder | Vector count per doc |
|---|---|---|---|---|---|
| Brazil tweets | ~55 PT-classified accounts (journalists, columnists, STF analysts) | 30-80 | Same as M3xA tweets, PT prefix | Cohere v4 1536d | 1 |
| Wire articles | BR-wire 1-6 (financial daily, mainstream daily, alt-political, popular tabloid, conservative, news network) | 400-2000 | Same as M3xA wires | Cohere v4 1536d | 1 or 1+N |
| Web columnists | Columnist portals, polling scanners, political analysis digests | 800-2500 | Same as M3xA long-form, often columnar structure | Cohere v4 1536d | 1+N |
| Email research | Bank A desk (BR), EM-desk BR (political), EM-desk BR (sentinel), Indie analyst BR (STF), Indie analyst BR (dialogues) | 1500-5000 | Section-aware | Cohere v4 1536d | N |
| Drive PDFs | Bank A-F (Brazilian banks + central bank analyses) | 3K-15K | Same triple-layer as M3xA PDFs | Cohere v4 1536d both modes | 1 + N + P |
| Calendar | Central bank SGS, Focus, COPOM | structured | DB rows | n/a | n/a |
| Markets | B3 futures, USDBRL, Ibov, DI1, PTAX | structured | DB rows | n/a | n/a |

### 3.3 AI domain (`ai_knowledge`, `mxai_knowledge`)

Already separate tables. Same strategy applied: Cohere v4 1536d, hierarchical for long blog posts and arXiv PDFs, 1-vector for short notes and tweets. No change to ingestion sources (15 RSS feeds, ArXiv keyword-filtered, mxai Twitter/code scrapers).

---

## 4 · Universal layer: Anthropic Contextual Retrieval

Applied to **every** chunk across **every** source type before embedding.

### 4.1 Prefix template

```
This chunk is from {source_type} by {entity_name} on {YYYY-MM-DD},
{section_heading_if_any}, discussing {1-line topic summary}.

{chunk_text}
```

Generated by Haiku at ingestion time. Stored separately in the row (for debugging / re-embedding) but prepended to chunk_text at embed time.

### 4.2 Haiku prompt (Anthropic-published template)

```
<document>
{WHOLE_DOCUMENT}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{CHUNK_CONTENT}
</chunk>

Please give a short succinct context to situate this chunk within
the overall document for the purposes of improving search retrieval
of the chunk. Answer only with the succinct context and nothing else.
```

### 4.3 Measured impact (Anthropic, 2024)

- Contextual embeddings alone: 35% reduction in retrieval failures (5.7% → 3.7%)
- + Contextual BM25: 49% reduction (5.7% → 2.9%)
- + Reranking: 67% reduction (5.7% → 1.9%)

This is the single highest-ROI intervention in the architecture. Not optional.

### 4.4 Cost at scale

~2500 chunks/day × ~1K tok prompt × Haiku pricing = ~$2.50/day. Trivial.

---

## 5 · Special case: Bank PDF triple-layer pattern

### 5.1 Motivation

Bank PDFs are the highest-density signal source in the corpus. Today: 1 vector per ~25K-token PDF. A query like "what did Bank 4 say about Brazil rates in Q3" cannot retrieve the specific Brazil-rates passage — only the whole PDF if dominant theme matches.

### 5.2 The three layers

**Layer A — Document-level vector.** Whole PDF (text-extracted) → Cohere v4 (text mode, 128K context handles any realistic note). One vector per PDF. Includes doc-level Haiku context prefix.

*Used for:* "which Bank N notes mentioned X" macro-routing queries.

**Layer B — Hierarchical text chunks.** Parent 2000 tok / child 512 tok / parent overlap 200 / child overlap 100. Section-aware splitting (markdown headers if PDF was structurally parsed, else recursive). Each child gets its own Haiku context prefix referencing doc title + date + section.

*Used for:* "what did Bank 4 specifically say about X" passage-level queries. LLM receives parent as context.

**Layer C — Page-image vectors.** Each PDF page rendered as image → Cohere v4 (image mode). One vector per page. Includes page metadata (page number, doc id, section heading where detected) as siblings, not in the embedding itself.

*Used for:* chart / table / equity-curve / scatter-plot queries that text extraction destroys. Cohere v4's ViDoRe benchmark performance targets exactly this case.

### 5.3 Retrieval-time fusion

```
Query → parallel:
  ├─ Layer A search (top 20)
  ├─ Layer B child search (top 100, parents returned)
  └─ Layer C image search (top 20)
                 │
                 ▼
       RRF fusion (k=60) → top 150
                 │
                 ▼
       Cohere Rerank 3.5 → top 30
                 │
                 ▼
       Synthesis context assembly
```

### 5.4 Storage impact

~50 active PDFs ingested per week × ~25 pages × ~5 children + 1 doc-vec = ~6500 vectors/week added. Cumulative ~340K vectors/year for PDF corpus alone. ~2GB at 1536d. Negligible.

---

## 6 · Special case: Podcast dual text+audio embedding

### 6.1 Motivation

Today: 1 vector per episode = retrieval black hole. A query about minute 47 of a 2-hour episode only hits if full-episode centroid happens to lean there.

### 6.2 The two paths

**Path 1 — Text (existing pipeline, chunked).** Groq Whisper transcribes → Haiku cleans → semantic chunker splits on cosine-drop topic boundaries → ~600 tok chunks (~5 min audio) with 100 tok overlap (~1 min). Speaker-turn aware where pyannote diarization succeeded. Each chunk carries `t_start_sec`, `t_end_sec`, `speaker`. Embedded with Cohere v4 1536d + contextual prefix.

**Path 2 — Audio (new, parallel).** Raw audio → Nova auto-segments at 30 sec → embed each segment. Each Nova embedding carries `episode_id`, `segment_idx`, `t_start_sec`, `t_end_sec`. Embedded with Nova 3072d. No prefix (audio modality).

### 6.3 Why both, not either

- Text path enables exact-quote retrieval, citation, and synthesis context (Sonnet needs text to read).
- Audio path enables semantic-audio retrieval that survives transcription noise, captures speaker tone shifts, and supports "moments where the tone changed" queries that text can never serve.
- They retrieve to the same `episode_id` + `t_start_sec` keys, so post-retrieval fusion is straightforward.

### 6.4 Citation pattern

Result citations resolve to `{episode_title} @ {mm:ss}`, dual-tagged with which path found it (for debugging during eval).

---

## 7 · Storage architecture

### 7.1 Tables (replacing current `unified_feed`)

```
unified_v5_text                      (Cohere v4 @ 1536d)
  ├ domain in (macro, brazil, ai)
  ├ source_type in (tweet, wire, longform, email, web,
  │                  pdf_doc, pdf_text_parent, pdf_text_child,
  │                  podcast_text, geo, ai_blog, ai_paper)
  ├ chunk_role in (whole, parent, child)
  ├ parent_doc_id, chunk_idx, n_chunks
  ├ section_heading, t_start_sec, t_end_sec (where applicable)
  ├ entity_id (resolved at ingest)
  ├ contextual_prefix (stored separately for debug / re-embed)
  └ source_id, created_at, priority_mult, scope_mult

unified_v5_pdf_images                (Cohere v4 @ 1536d, image mode)
  ├ domain in (macro, brazil)
  ├ source_type = pdf_page_image
  ├ parent_doc_id, page_number, n_pages
  ├ section_heading_detected
  ├ entity_id
  └ source_id, created_at

unified_v5_audio                     (Nova @ 3072d)
  ├ domain in (macro, brazil)
  ├ source_type = podcast_audio
  ├ episode_id, segment_idx, n_segments
  ├ t_start_sec, t_end_sec
  ├ speaker (when diarized)
  └ source_id, created_at
```

Existing structured DBs unchanged:
- `economic_calendar_*.db`
- `polymarket.db`
- `timeline_events.db`
- `events_timeline_macro.db` / `events_timeline_brazil.db`
- `golden_exchanges.db`
- `source_classifications.db`
- `telegram_conversations.db`
- `brainstorm_sessions.db`

### 7.2 LanceDB compaction

Existing 6-hourly compaction cron (`lancedb_compact · 0 0,4,8,12,16,20`) extends to cover the three new tables. No new cron logic.

### 7.3 BM25 indexes (unchanged from Phase 1 plan)

Two per-language Tantivy FTS indexes on `unified_v5_text.contextual_prefix + chunk_text`:
- `en_stem` for `domain in (macro, ai)`
- `pt_stem` for `domain = brazil`

The contextual prefix being indexed by BM25 is the contextual-BM25 part of Anthropic's technique — it's what takes the lift from 35% to 49%.

---

## 8 · Hybrid retrieval flow with this stack

Extends the predecessor hybrid-retrieval pipeline stages, replacing the embedder/reranker:

```
              QUERY
                │
                ▼
      ┌───────────────────────────┐
      │ ① Query rewriter (Haiku)  │
      │ → 2-4 sub-queries         │
      │ → rerank instruction      │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌───────────────────────────┐
      │ ② Semantic cache check    │
      │ (sqlite-vec, local)       │
      └─────────────┬─────────────┘
                    │ MISS
                    ▼
   ┌────────────────┴────────────────────┐
   │ PARALLEL per sub-query              │
   ├─────────────────────────────────────┤
   │ ③a LanceDB FTS (BM25)               │
   │    en_stem + pt_stem on             │
   │    unified_v5_text → top 200        │
   │ ③b LanceDB vector search:           │
   │    - unified_v5_text          (200) │
   │    - unified_v5_pdf_images    (50)  │
   │    - unified_v5_audio         (50)  │
   │ ③c Entity-KG 2-hop traversal (100)  │
   └────────────────┬────────────────────┘
                    │
                    ▼
      ┌───────────────────────────┐
      │ ④ RRF fusion (k=60)       │
      │   → top 150               │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌───────────────────────────┐
      │ ⑤ Cohere Rerank 3.5       │
      │   (Bedrock API call,      │
      │   instruction from ①)     │
      │   → top 40                │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌───────────────────────────┐
      │ ⑥ Editorial layer         │
      │   (multipliers + entity   │
      │   floor) → top 30         │
      └─────────────┬─────────────┘
                    │
                    ▼
        Parent-doc expansion for child hits
        (Layer B PDFs return parent; podcast
        chunks return ±1 chunk window)
                    │
                    ▼
        Context assembly → Sonnet synth
```

T1 simple queries still bypass stages ① and ⑤ via the existing Haiku classifier router.

---

## 9 · M3xA vs M3xBr — what differs

**Embedding layer: nothing differs.** Cohere v4's unified multilingual space holds EN + PT + AR content in one table. Verified via Cohere Embed v4 release notes: "All languages map to same dimensional space, enabling cross-lingual search."

**What differs:**

1. **BM25 indexes per language** (en_stem, pt_stem). Lexical tokenization is language-dependent; embeddings aren't.
2. **Source mix and weighting** — separate `source_classifications.db` rows per domain, separate priority multipliers per soul file.
3. **Routing at query time** — bot identity sets the `domain` filter, same as today.
4. **Cross-lingual queries become native** — a query in English can hit Portuguese content and vice versa. Exposed by default; no `#cross-lingual` flag needed.

**Entity registry is shared** — entity files don't care about language. PT entities (Brazilian journalists, central bankers, columnists) already live in the same registry as EN.

---

## 10 · Advanced techniques — what's in v5, what's deferred

| Technique | Status | Rationale |
|---|---|---|
| Anthropic Contextual Retrieval | **In** (universal) | +35-67% retrieval lift, applies everywhere, cheap |
| Hierarchical parent-child chunking | **In** | Native to Bedrock KB, biggest single win for long docs |
| Cohere Rerank 3.5 | **In** | Finance SOTA, multilingual, drop-in via Bedrock |
| Hybrid dense + BM25 (RRF) | **In** | Already planned; en_stem + pt_stem Tantivy |
| Multimodal PDF page-image embeddings | **In** (Phase 5) | Cohere v4 makes it trivially cheap; bank PDFs justify it alone |
| Dual text+audio podcast embeddings | **In** (Phase 6) | Nova GA Oct 2025, no infra burden, fixes known retrieval hole |
| Semantic (topic-shift) chunking | **In** (podcasts, long-form) | Cost negligible at scale |
| Late chunking (Jina-style) | **Deferred** | Bedrock APIs don't expose token-level outputs. Would need self-host. Spark backlog. |
| Agentic chunking (LLM picks boundaries per doc) | **Deferred** | Still experimental; complexity high, marginal lift over hierarchical. Revisit Q4 2026. |
| Propositions-based chunking | **Deferred** | Strong for legal/clinical, marginal for news/research. Revisit if eval shows it. |
| Multi-scale indexing (AI21's RRF over chunk sizes) | **Phase 7 optional** | +1-37% lift but ~3× storage. Add only if Phases 1-6 plateau. |
| ColBERT / late-interaction | **Not on Bedrock** | Self-host required. Stays Spark backlog. |
| Entity-KG 2-hop traversal | **In** (already in Phase 1 of the predecessor hybrid-retrieval design) | Independent of chunking |
| Self-corrective agentic RAG loop | **Future** | Retrieval-quality intervention; layer on after Phase 7 |

---

## 11 · Cost analysis

At current volumes (~500 docs/day, ~10 podcasts/day, ~50 PDFs/week, ~100 queries/day):

| Item | Volume | Unit cost | Daily | Monthly |
|---|---|---|---|---|
| Cohere v4 embed (text) | ~2.5M tok | $0.12/M | $0.30 | $9 |
| Cohere v4 embed (PDF images) | ~50 PDFs × 25 pages | per image | $0.10 | $3 |
| Nova embed (audio) | ~2400 segments | per segment | $0.05 | $1.50 |
| Haiku contextual prefix | ~2500 chunks × 1K prompt tok | Haiku rate | $2.50 | $75 |
| Cohere Rerank 3.5 | 100 queries × 150 docs | per call | $0.20 | $6 |
| **Total ongoing** | | | **~$3.15** | **~$95** |
| Initial backfill (one-time) | ~3B tokens | $0.12/M | — | ~$360 |

Trivial relative to system value. Cost is genuinely not a constraint here.

---

## 12 · Phased rollout

### Phase 0 — Eval foundation (gate to everything)
- [ ] Build 200-query golden set from `telegram_conversations.db`, covering all source classes + known failure modes from the internal lessons log (#15, #21, #22, #44)
- [ ] Add PT-language queries (~30%) and cross-lingual queries (~10%)
- [ ] Add PDF-specific queries that hit charts/tables (Layer C target)
- [ ] Add podcast-specific queries with timestamp expectations
- [ ] Score current Voyage-3-large baseline → relevance@10, MRR, did-right-source-appear-in-top-30
- [ ] Document baseline numbers in the evaluator rubric document

### Phase 1 — Cohere v4 swap, no chunking change
- [ ] Provision Bedrock access for `cohere.embed-v4` in us-east-1
- [ ] Stand up `unified_v5_text` table on LanceDB
- [ ] Backfill 30 days of current `unified_feed` rows into `unified_v5_text` with Cohere v4 1536d (1-vector-per-doc, no chunking, no prefix yet)
- [ ] Wire `RETRIEVAL_MODE=v5_bedrock` env var (current path stays default)
- [ ] Shadow eval vs Voyage baseline on golden set
- [ ] Gate: ≥5% relevance lift, no PT regression
- [ ] If gated: cut over default for shadow traffic 10%

### Phase 2 — Hierarchical chunking for long docs
- [ ] Implement chunker as Bedrock Knowledge Base custom Lambda OR as ingestion-side Python (decide based on KB vs direct LanceDB tradeoffs)
- [ ] Apply hierarchical to: longform articles, email research notes, PDF text (Layer B), web columnists
- [ ] Tweets and short wires stay 1-vector
- [ ] Backfill long-doc subset
- [ ] Shadow eval
- [ ] Gate: long-doc query lift visible

### Phase 3 — Anthropic Contextual Retrieval prefix
- [ ] Implement Haiku context generator (per chunk, per ingestion event)
- [ ] Apply prefix to every chunk before embed AND before BM25 index
- [ ] Re-embed everything in `unified_v5_text` with prefixes
- [ ] Rebuild en_stem + pt_stem Tantivy indexes over `contextual_prefix + chunk_text`
- [ ] Shadow eval
- [ ] Gate: Anthropic's 35-49% lift measurable on golden set

### Phase 4 — Cohere Rerank 3.5 wired
- [ ] Provision Bedrock access for `cohere.rerank-v3-5`
- [ ] Wire at stage ⑤ of hybrid pipeline (instruction emitted by Haiku rewriter)
- [ ] Top-150 → top-40 → top-30 after editorial layer
- [ ] Shadow eval
- [ ] Gate: compounded lift ~50-67% per Anthropic data
- [ ] Cut over default fully if eval clean

### Phase 5 — Multimodal PDF page-image layer
- [ ] Stand up `unified_v5_pdf_images` table
- [ ] PDF page renderer (pdf2image or pypdf-poppler) on ingestion path
- [ ] Embed each page with Cohere v4 image mode → store
- [ ] Wire as third leg of stage ③b
- [ ] Backfill Bank 1-11 + Bank A-F PDFs from last 90 days
- [ ] Eval: chart/table-specific query subset of golden set
- [ ] Gate: chart queries show retrieval that text-only path missed

### Phase 6 — Dual podcast embeddings (Nova audio path)
- [ ] Provision Bedrock access for `amazon.nova-multimodal-embed-v1`
- [ ] Stand up `unified_v5_audio` table
- [ ] Modify podcast ingestion to also send raw audio to Nova (segment-by-segment via auto-segmentation)
- [ ] Wire as fourth leg of stage ③b
- [ ] Backfill last 60 days of podcast audio
- [ ] Eval: podcast-specific query subset
- [ ] Gate: semantic-audio queries work; timestamps resolve correctly

### Phase 7 (optional) — Multi-scale indexing
- [ ] Only if Phase 6 evals plateau and golden-set ceiling is being hit
- [ ] Index same chunks at 256 / 512 / 1024 tok parallel
- [ ] RRF fuse across scales (per AI21 pattern)
- [ ] Eval ROI vs ~3× storage cost

---

## 13 · Open questions

- **Bedrock Knowledge Base vs direct LanceDB+Bedrock API**: KB handles hierarchical chunking and contextual prefix natively but locks ingestion path to S3 → KB → OpenSearch/S3-Vectors. Direct API preserves the current LanceDB architecture. Phase 1 spike: prototype both, pick based on ops complexity vs feature coverage.
- **Parent-doc storage location**: parents in LanceDB rows (filterable on metadata, but heavier) or in a sibling KV store (Redis / Postgres)?
- **PDF page rendering pipeline**: pdf2image (poppler) vs Bedrock Data Automation. BDA handles parsing + extraction but adds another Bedrock service. Phase 5 decision.
- **Cross-lingual query routing**: should the global-domain bot return Brazil-domain hits by default, or behind an opt-in flag? UX call, not architectural.
- **Re-embedding cadence on prompt/prefix changes**: if we change the contextual prefix template, do we re-embed everything? Yes — but that's a deliberate migration, not a hot-path concern.
- **Nova region availability**: GA in us-east-1; cross-region inference status TBD. Verify at Phase 6 entry.
- **Speaker diarization quality on multi-speaker podcasts**: pyannote degrades at 4+ speakers. Affects timestamp metadata quality but not embeddings themselves.
- **Reranker instruction template per query type**: should this come from the soul-corrections deltas? Closes loop between the self-evaluator and retrieval.

---

## 15 · Reference

### External — research that drove this spec
- Anthropic: "Contextual Retrieval" (Sep 2024). 35-67% retrieval failure reduction.
- AWS: "Contextual retrieval in Anthropic using Amazon Bedrock Knowledge Bases" (Jun 2025). Reference implementation.
- Cohere: Embed v4 release (April 2025). Multimodal, multilingual, 128K context, finance-tuned.
- Amazon: Nova Multimodal Embeddings (Oct 2025 GA). Native audio embeddings, 200+ languages.
- Cohere: Rerank 3.5 release (Dec 2024). +23.4% NDCG@10 vs Hybrid Search on finance.
- Jina AI: "Late Chunking" arXiv 2409.04701 (Sep 2024, updated Jul 2025). Deferred to Spark backlog.
- AI21: "Multi-scale chunking" (Jan 2026). Phase 7 optional.
- Greg Kamradt: "5 Levels of Text Splitting" framework.
- Vecta (Feb 2026 benchmark): recursive 512-token splitting wins at 69% accuracy on academic papers.
- Chroma "context rot" research (Jul 2025): generation degrades past ~2500 tok of context.
- MMTEB / MTEB leaderboard: multilingual benchmarking framework.
- FinanceBench / FinMTEB: financial-domain evaluation suites for future fine-tune consideration.
