# Understanding SemEval Metrics: A Deep Dive

> A comprehensive technical guide to understanding, interpreting, and applying evaluation metrics in semantic embeddings.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Information Retrieval Metrics](#information-retrieval-metrics)
3. [Semantic Similarity Metrics](#semantic-similarity-metrics)
4. [Linguistic Robustness Metrics](#linguistic-robustness-metrics)
5. [Vector Arithmetic Metrics](#vector-arithmetic-metrics)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)

---

## Introduction

Evaluating semantic embeddings is a multifaceted challenge. Unlike traditional classification tasks where accuracy is often sufficient, semantic embeddings require metrics that capture:

- **Ranking quality**: How well does the model rank relevant items?
- **Semantic understanding**: Does the model capture meaning relationships?
- **Robustness**: How stable are embeddings under variations?
- **Compositional reasoning**: Can the model perform vector arithmetic?

SemEval provides **4 task categories** with **20+ metrics** to comprehensively evaluate these aspects. This guide explains each metric in depth.

---

## Information Retrieval Metrics

Information Retrieval (IR) metrics evaluate how well a model retrieves relevant documents for given queries. These are ranking-based metrics that consider both **relevance** and **position** in the ranked list.

### Core Concepts

**Relevance Scores**: In SemEval, documents are judged on a 3-level scale:
- `0`: Not relevant
- `1`: Partially relevant (topically related but not ideal)
- `2`: Highly relevant (exactly what the query asks for)

This graded relevance allows for more nuanced evaluation than binary relevant/non-relevant.

---

### NDCG@k (Normalized Discounted Cumulative Gain)

**What it measures**: Ranking quality with graded relevance, emphasizing top positions.

**Why it matters**: In real-world search, users primarily look at top results. NDCG penalizes relevant documents appearing lower in the ranking.

#### Mathematical Definition

```
DCG@k = Σ(i=1 to k) [rel_i / log2(i + 1)]
NDCG@k = DCG@k / IDCG@k
```

Where:
- `rel_i`: Relevance score of document at position i
- `IDCG@k`: Ideal DCG (if all documents were perfectly ranked)

#### Interpretation

| NDCG@10 | Quality | What it means |
|---------|---------|---------------|
| 0.9-1.0 | Excellent | Nearly perfect ranking |
| 0.7-0.9 | Good | Relevant docs mostly at top |
| 0.5-0.7 | Fair | Some relevant docs buried |
| 0.0-0.5 | Poor | Poor ranking quality |

#### Example

**Query**: "machine learning algorithms"

**Retrieved (top 5)**:
1. "Introduction to ML Algorithms" (score: 2) → 2.0 / log2(2) = 2.0
2. "Deep Learning Guide" (score: 1) → 1.0 / log2(3) = 0.631
3. "Neural Networks Tutorial" (score: 1) → 1.0 / log2(4) = 0.5
4. "Cooking Recipes" (score: 0) → 0.0
5. "ML Research Papers" (score: 2) → 2.0 / log2(6) = 0.774

DCG@5 = 2.0 + 0.631 + 0.5 + 0.0 + 0.774 = **3.905**

Ideal ranking would place both score=2 docs first:
IDCG@5 = 2.0 + 2.0/log2(3) + 1.0/log2(4) + 1.0/log2(5) + 0.0 = **4.762**

NDCG@5 = 3.905 / 4.762 = **0.820** (Good!)

**Key Insight**: Even though we got high-quality results, the model placed "ML Research Papers" at position 5 instead of 2, reducing NDCG.

---

### MRR@k (Mean Reciprocal Rank)

**What it measures**: How quickly users find the first relevant result.

**Why it matters**: For many queries, users stop searching after finding one good result. MRR captures this "success at first encounter" behavior.

#### Mathematical Definition

```
RR = 1 / rank_of_first_relevant_item
MRR = average(RR) across all queries
```

#### Interpretation

| MRR@10 | Quality | What it means |
|--------|---------|---------------|
| 0.8-1.0 | Excellent | Relevant doc usually in top 2 |
| 0.5-0.8 | Good | Relevant doc in top 3-5 |
| 0.2-0.5 | Fair | First relevant doc often buried |
| 0.0-0.2 | Poor | Rarely finds relevant docs early |

#### Example

**Query 1**: First relevant doc at position 1 → RR = 1.0
**Query 2**: First relevant doc at position 3 → RR = 0.333
**Query 3**: First relevant doc at position 2 → RR = 0.5

MRR = (1.0 + 0.333 + 0.5) / 3 = **0.611**

**Key Insight**: MRR only cares about the first relevant result. It's harsh but realistic for navigational queries.

---

### MAP@k (Mean Average Precision)

**What it measures**: Precision at every relevant document position, averaged.

**Why it matters**: Unlike MRR (which only cares about first result), MAP rewards systems that place ALL relevant documents high in the ranking.

#### Mathematical Definition

```
AP@k = (1/R) × Σ [Precision@i × rel(i)]
MAP@k = average(AP@k) across all queries
```

Where:
- `R`: Total number of relevant documents
- `rel(i)`: 1 if doc at position i is relevant, 0 otherwise
- `Precision@i`: Precision at position i

#### Interpretation

| MAP@10 | Quality | What it means |
|--------|---------|---------------|
| 0.8-1.0 | Excellent | Most relevant docs at top |
| 0.6-0.8 | Good | Relevant docs generally high |
| 0.4-0.6 | Fair | Mixed relevant/irrelevant |
| 0.0-0.4 | Poor | Relevant docs scattered |

#### Example

**Query**: "python tutorials" (3 relevant docs in corpus)

**Retrieved (top 10)**:
1. Relevant → P@1 = 1/1 = 1.0
2. Irrelevant → (no contribution)
3. Relevant → P@3 = 2/3 = 0.667
4. Irrelevant → (no contribution)
5. Relevant → P@5 = 3/5 = 0.6
6-10. Irrelevant → (no contribution)

AP@10 = (1/3) × (1.0 + 0.667 + 0.6) = **0.756**

**Key Insight**: MAP penalizes irrelevant documents interspersed between relevant ones. It's stricter than MRR but more forgiving than NDCG about specific positions.

---

### Precision@k and Recall@k

**What they measure**: Standard IR metrics adapted to top-k results.

#### Precision@k

```
Precision@k = (# relevant docs in top-k) / k
```

**Example**: Out of top 10 results, 7 are relevant → P@10 = 0.7

**Use case**: "What fraction of retrieved documents are actually relevant?"

#### Recall@k

```
Recall@k = (# relevant docs in top-k) / (total relevant docs)
```

**Example**: Corpus has 20 relevant docs, you found 8 in top-10 → R@10 = 0.4

**Use case**: "What fraction of all relevant documents did we retrieve?"

**Trade-off**: High k → High recall, low precision. Low k → Low recall, high precision.

---

### When to Use Which IR Metric?

| Use Case | Recommended Metric | Why? |
|----------|-------------------|------|
| Search engines (web search) | NDCG@10 | Users scan top results, graded relevance matters |
| Question answering | MRR@5 | Users want one good answer quickly |
| Document discovery | MAP@20 | Need to find ALL relevant documents |
| Quick lookups | Precision@3 | Only care about top few results |
| Comprehensive retrieval | Recall@50 | Need to ensure nothing is missed |

---

## Semantic Similarity Metrics

Semantic similarity metrics evaluate whether the model understands meaning relationships using **triplet evaluation**.

### The Triplet Paradigm

A triplet consists of:
- **Anchor** (A): Reference text
- **Positive** (P): Semantically similar to anchor
- **Negative** (N): Semantically dissimilar to anchor

**Success criterion**: `similarity(A, P) > similarity(A, N)`

This tests if the embedding space correctly clusters similar concepts together.

---

### Triplet Accuracy

**What it measures**: Percentage of triplets where positive is closer to anchor than negative.

#### Mathematical Definition

```
Accuracy = (# correct triplets) / (total triplets)

Correct if: cosine_sim(A, P) > cosine_sim(A, N)
```

#### Interpretation

| Accuracy | Quality | What it means |
|----------|---------|---------------|
| 0.95-1.0 | Excellent | Nearly perfect semantic understanding |
| 0.85-0.95 | Good | Strong semantic clustering |
| 0.70-0.85 | Fair | Decent but some confusion |
| 0.0-0.70 | Poor | Poor semantic discrimination |

#### Example

**Triplet 1**:
- Anchor: "The stock market crashed today"
- Positive: "Equity prices fell sharply"
- Negative: "It's raining outside"

sim(A, P) = 0.87, sim(A, N) = 0.12 → ✅ Correct

**Triplet 2**:
- Anchor: "Machine learning model"
- Positive: "AI algorithm"
- Negative: "Neural network"

sim(A, P) = 0.92, sim(A, N) = 0.94 → ❌ Wrong!

The model thinks "neural network" is MORE similar to "machine learning model" than "AI algorithm". While defensible, this violates our triplet assumption.

Accuracy = 1/2 = **0.50**

**Key Insight**: Triplet accuracy reveals if your model's notion of similarity aligns with human judgment.

---

### Margin Metrics

**What they measure**: How confident the model is in distinguishing positive from negative.

#### Average Margin

```
Margin = sim(A, P) - sim(A, N)
Avg_Margin = mean(margins across all triplets)
```

**Interpretation**:
- **Margin > 0.3**: Strong confidence, clear distinction
- **Margin 0.1-0.3**: Moderate confidence
- **Margin 0-0.1**: Weak confidence, borderline cases
- **Margin < 0**: Wrong prediction

#### Example

**High-quality model**:
- Triplet 1: 0.92 - 0.23 = **0.69** (very confident)
- Triplet 2: 0.88 - 0.31 = **0.57** (very confident)
- Triplet 3: 0.79 - 0.68 = **0.11** (weak but correct)
- Avg Margin = **0.457**

**Low-quality model**:
- Triplet 1: 0.71 - 0.68 = **0.03** (barely correct)
- Triplet 2: 0.83 - 0.86 = **-0.03** (wrong!)
- Triplet 3: 0.66 - 0.55 = **0.11** (weak)
- Avg Margin = **0.037**

**Key Insight**: Even if accuracy is same, margin tells you HOW confident the model is. High margin = robust predictions.

---

### Margin Distribution

**What it measures**: Breakdown of how many triplets have strong vs. weak margins.

#### Metrics

- **margin_gt_01**: % of triplets with margin > 0.1
- **margin_gt_02**: % of triplets with margin > 0.2

#### Interpretation

Ideal distribution (robust model):
- 95% with margin > 0.1
- 85% with margin > 0.2

Poor distribution:
- 60% with margin > 0.1
- 30% with margin > 0.2

**Use case**: Diagnose if your model is "barely correct" vs. "confidently correct".

---

### Category Breakdown

**What it measures**: Performance stratified by semantic category (e.g., finance, sports, technology).

**Why it matters**: Models often excel in some domains but struggle in others.

#### Example Output

```
Category Breakdown:
  finance:  accuracy=0.92, avg_margin=0.45, count=120
  sports:   accuracy=0.88, avg_margin=0.38, count=85
  medicine: accuracy=0.67, avg_margin=0.12, count=50  ← Problem area!
```

**Interpretation**: This model is weak at medical domain semantics. You might need:
- More medical training data
- Domain-specific fine-tuning
- Different embedding model

---

## Linguistic Robustness Metrics

Robustness metrics evaluate how **stable** embeddings are under linguistic variations. Unstable embeddings hurt production systems.

### Why Robustness Matters

Real-world text is messy:
- Users make typos
- Languages have inflections (plural, tense, gender)
- Negation changes meaning dramatically

**Goal**: Embeddings should handle expected variations gracefully while catching semantic changes.

---

### Morphology Robustness

**What it tests**: Stability under morphological changes (plural, tense, case, etc.)

**Expectation**: Morphological variants should have **HIGH similarity** (>0.85) to originals.

#### Mathematical Definition

```
Morphology_Success_Rate = (# pairs with sim >= 0.85) / (total pairs)
```

#### Example Test Cases

| Original | Variant | Type | Expected Similarity |
|----------|---------|------|---------------------|
| "car" | "cars" | Plural | > 0.85 |
| "run" | "running" | Gerund | > 0.85 |
| "happy" | "happier" | Comparative | > 0.85 |
| "cat" | "cats" | Plural | > 0.85 |

#### Good vs. Bad Model

**Robust Model**:
```
"book" ↔ "books":    sim=0.94  ✓
"run" ↔ "running":   sim=0.91  ✓
"fast" ↔ "faster":   sim=0.88  ✓
Success Rate: 100%
```

**Brittle Model**:
```
"book" ↔ "books":    sim=0.72  ✗ (treats as different words!)
"run" ↔ "running":   sim=0.68  ✗
"fast" ↔ "faster":   sim=0.81  ✗
Success Rate: 0%
```

**Key Insight**: Low morphology robustness means your search/retrieval will fail on simple variations. "looking for a car" won't match "we sell cars".

---

### Typo Robustness

**What it tests**: Stability under spelling errors.

**Expectation**: Typos should have **HIGH similarity** (>0.75) to correct spellings.

#### Mathematical Definition

```
Typo_Success_Rate = (# pairs with sim >= 0.75) / (total pairs)
```

#### Example Test Cases

| Correct | Typo | Type | Expected Similarity |
|---------|------|------|---------------------|
| "computer" | "compter" | Missing letter | > 0.75 |
| "restaurant" | "restarant" | Transposition | > 0.75 |
| "definitely" | "definately" | Common error | > 0.75 |

#### Why Threshold is 0.75 (Not 0.85)?

Typos change **form** more than morphology:
- Morphology: "run" → "running" (systematic, expected)
- Typo: "run" → "rnu" (random, noisy)

We allow more tolerance (0.75 vs. 0.85) but still expect robustness.

#### Typo Type Breakdown

```python
Type Breakdown:
  keyboard_adjacent:  success_rate=0.92  # e.g., "teh" → "the"
  phonetic:          success_rate=0.78  # e.g., "phone" → "fone"
  missing_letter:    success_rate=0.85  # e.g., "letter" → "leter"
  doubled_letter:    success_rate=0.88  # e.g., "runing" → "running"
```

**Use case**: Identify which typo types your model struggles with most.

---

### Negation Robustness

**What it tests**: Detection of meaning changes from negation.

**Expectation**: Negated text should have **LOW similarity** (<0.50) to originals.

#### Mathematical Definition

```
Negation_Success_Rate = (# pairs with sim < 0.50) / (total pairs)
```

#### Example Test Cases

| Original | Negated | Expected Similarity |
|----------|---------|---------------------|
| "I like this movie" | "I don't like this movie" | < 0.50 |
| "The product works" | "The product doesn't work" | < 0.50 |
| "Good quality" | "Not good quality" | < 0.50 |

#### Good vs. Bad Model

**Semantic-aware Model**:
```
"good" ↔ "not good":           sim=0.32  ✓ (recognizes opposition)
"works" ↔ "doesn't work":      sim=0.28  ✓
"I agree" ↔ "I don't agree":   sim=0.41  ✓
Success Rate: 100%
```

**Bag-of-words Model** (naive):
```
"good" ↔ "not good":           sim=0.82  ✗ (only sees "good")
"works" ↔ "doesn't work":      sim=0.89  ✗
"I agree" ↔ "I don't agree":   sim=0.91  ✗
Success Rate: 0%
```

**Key Insight**: Many simple models fail negation tests because they focus on content words and ignore negation markers. This is critical for sentiment analysis and opinion mining.

---

### Overall Robustness Score

**What it measures**: Weighted average of all robustness subtasks.

#### Mathematical Definition

```
Overall_Robustness = (
    morphology_success × morphology_count +
    typo_success × typo_count +
    negation_success × negation_count
) / (total_count)
```

#### Interpretation

| Score | Quality | What it means |
|-------|---------|---------------|
| 0.90-1.0 | Excellent | Production-ready robustness |
| 0.75-0.90 | Good | Mostly stable, minor issues |
| 0.60-0.75 | Fair | Significant stability problems |
| 0.0-0.60 | Poor | Brittle, unreliable in production |

---

## Vector Arithmetic Metrics

Vector arithmetic tests **compositional reasoning**: can the model perform semantic algebra?

### The Analogy Task

**Format**: "A is to B as C is to D"

**Vector Operation**: `D ≈ A - B + C`

#### Example

"Paris" is to "France" as "Berlin" is to ???

```
embedding("Germany") ≈ embedding("Berlin") - embedding("Paris") + embedding("France")
```

**Why it matters**: This tests if the embedding space has meaningful geometric structure where semantic relationships are linear transformations.

---

### Top-k Accuracy

**What it measures**: Whether the correct answer appears in the top-k predictions.

#### Mathematical Definition

```
Accuracy@k = (# analogies where correct answer in top-k) / (total analogies)
```

#### Interpretation

| Metric | Good Model | Poor Model |
|--------|------------|------------|
| Accuracy@1 | 0.60 | 0.10 |
| Accuracy@5 | 0.85 | 0.25 |
| Accuracy@10 | 0.92 | 0.40 |

**Key Insight**: Gap between @1 and @10 reveals if model "knows" the answer but lacks confidence to rank it first.

---

### Mean Rank

**What it measures**: Average position of correct answer.

#### Mathematical Definition

```
Mean_Rank = average(rank of correct answer across all analogies)
```

#### Interpretation

- **Rank 1-2**: Excellent (correct answer usually first)
- **Rank 3-5**: Good (correct answer in top few)
- **Rank 6-20**: Fair (correct answer buried)
- **Rank > 20**: Poor (rarely finds correct answer)

#### Example

```
Analogy 1: "king" - "man" + "woman" = "queen" → Rank 1
Analogy 2: "Paris" - "France" + "Germany" = "Berlin" → Rank 3
Analogy 3: "up" - "down" + "left" = "right" → Rank 2
Analogy 4: "cat" - "kitten" + "dog" = "puppy" → Rank 8

Mean Rank = (1 + 3 + 2 + 8) / 4 = 3.5
```

**Key Insight**: Lower is better. Mean rank gives a single number summarizing overall analogy performance.

---

### Mean Reciprocal Rank (MRR)

**What it measures**: Harmonic mean of ranks (emphasizes top positions).

#### Mathematical Definition

```
MRR = average(1/rank across all analogies)
```

#### Why MRR vs. Mean Rank?

MRR penalizes low ranks more heavily:

| Rank | Reciprocal Rank | Contribution to Mean Rank |
|------|-----------------|---------------------------|
| 1 | 1.0 | 1 |
| 2 | 0.5 | 2 |
| 5 | 0.2 | 5 |
| 10 | 0.1 | 10 |
| 100 | 0.01 | 100 |

**Example**: Two models both have 2 analogies:

**Model A**: Ranks [1, 100]
- Mean Rank: 50.5
- MRR: (1.0 + 0.01) / 2 = **0.505**

**Model B**: Ranks [10, 10]
- Mean Rank: 10
- MRR: (0.1 + 0.1) / 2 = **0.1**

Mean Rank says Model B is better (10 vs. 50.5), but MRR says Model A is better (0.505 vs. 0.1) because it got one analogy perfect.

**Use case**: Prefer MRR when you care more about "getting some analogies perfect" vs. "being consistently mediocre".

---

### Category Breakdown

**What it measures**: Performance stratified by analogy type.

#### Common Categories

- **Geography**: "Paris - France + Germany = Berlin"
- **Gender**: "king - man + woman = queen"
- **Temporal**: "yesterday - past + future = tomorrow"
- **Comparative**: "good - better + bad = worse"

#### Example Output

```
Category Breakdown:
  geography:   accuracy@5=0.92, mean_rank=2.1
  gender:      accuracy@5=0.85, mean_rank=3.4
  temporal:    accuracy@5=0.68, mean_rank=7.2  ← Weak!
  comparative: accuracy@5=0.71, mean_rank=6.8
```

**Interpretation**: This model excels at geographic relationships but struggles with temporal reasoning.

---

## Interpreting Results

### Holistic Model Assessment

A good semantic embedding model should:

1. **Retrieve relevant content** (high NDCG@10)
2. **Understand semantic similarity** (high triplet accuracy)
3. **Be robust to variations** (high robustness scores)
4. **Perform compositional reasoning** (decent analogy accuracy)

### Red Flags

| Issue | Symptoms | Likely Cause |
|-------|----------|--------------|
| High IR but low similarity | NDCG=0.9 but accuracy=0.6 | Model does keyword matching, not semantic |
| High similarity but low IR | Accuracy=0.95 but NDCG=0.5 | Good at pairs, bad at ranking many docs |
| Low morphology robustness | <0.7 | Subword tokenization issues |
| Low negation robustness | <0.5 | No negation handling in architecture |
| Low analogy accuracy | <0.3 @5 | Embedding space not linearly structured |

### Cross-Task Correlations

**Insight 1**: IR and Similarity scores often correlate
- Both test "understanding relevance"
- If one is low, investigate embedding quality

**Insight 2**: Robustness is independent of accuracy
- You can have high accuracy but low robustness (brittle model)
- Or low accuracy but high robustness (stable but weak)

**Insight 3**: Analogy performance predicts transfer learning
- Models good at analogies often transfer better to new domains
- Tests if embedding space has meaningful structure

---

## Best Practices

### 1. Choose Metrics Based on Use Case

| Application | Priority Metrics |
|-------------|------------------|
| Search engine | NDCG@10, MRR@5 |
| Chatbot retrieval | Triplet accuracy, MRR@3 |
| Duplicate detection | Triplet accuracy, margin > 0.2 |
| Cross-lingual transfer | Analogy accuracy, robustness scores |
| Production systems | ALL robustness metrics (critical!) |

### 2. Set Realistic Thresholds

Don't expect perfection:
- NDCG@10 > 0.7 is production-ready for most applications
- Triplet accuracy > 0.85 is good
- Analogy @5 accuracy > 0.6 is respectable

### 3. Investigate Failures

When metrics are low:

1. **Export failed examples** using SemEval's export tools
2. **Analyze patterns**: Do failures cluster by category? Difficulty?
3. **Check data quality**: Are labels correct? Is test set representative?
4. **Compare baselines**: Is your model actually better than simpler alternatives?

### 4. Track Metrics Over Time

```python
# Log metrics across training
epoch_1: NDCG=0.65, accuracy=0.78, robustness=0.82
epoch_5: NDCG=0.72, accuracy=0.84, robustness=0.79  ← Robustness decreased!
epoch_10: NDCG=0.78, accuracy=0.89, robustness=0.75 ← Still decreasing!
```

**Insight**: Model is overfitting to accuracy while becoming less robust. Need regularization.

### 5. Test on Multiple Languages/Domains

If possible:
- Evaluate on multiple languages (if using multilingual models)
- Test on in-domain and out-of-domain data
- Check if rankings change across test sets

---

## Conclusion

SemEval's metric suite provides a **comprehensive view** of embedding quality:

- **IR metrics** (NDCG, MRR, MAP) → Ranking quality
- **Similarity metrics** (Accuracy, Margin) → Semantic understanding
- **Robustness metrics** (Morphology, Typo, Negation) → Stability
- **Arithmetic metrics** (Analogy, Top-k) → Compositional reasoning

No single metric tells the whole story. Use SemEval to:
1. Benchmark models comprehensively
2. Identify specific weaknesses
3. Track improvements during development
4. Ensure production readiness

**Remember**: Metrics are tools for insight, not just numbers to maximize. Always validate with real-world usage!

---

**For implementation details, see**:
- `semeval/metrics/ir_metrics.py`
- `semeval/metrics/similarity_metrics.py`
- `semeval/metrics/robustness_metrics.py`
- `semeval/metrics/arithmetic_metrics.py`

**Made with ❤️ for better embeddings**
