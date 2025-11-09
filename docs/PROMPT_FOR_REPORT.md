# Prompt Template for Evaluation Report Generation

> This document provides a structured prompt template for generating comprehensive evaluation reports from SemEval test results.

---

## Purpose

This prompt template helps generate detailed, actionable evaluation reports that:
- Analyze semantic embedding model performance across multiple tasks
- Provide metric interpretations with real-world examples
- Identify strengths, weaknesses, and production blockers
- Offer concrete recommendations with timelines
- Present findings in a clear, professional format

---

## Input Requirements

Before generating a report, you need the following inputs:

### 1. Evaluation Results File

**Format**: Markdown or JSON
**Location**: `output/results.md` or `output/results.json`
**Contains**:
- Task execution status (success/failure)
- Runtime metrics
- Performance scores for each task
- Breakdowns by category, difficulty, subcategory

**Example**:
```markdown
# Evaluation Results
**Generated:** 2025-11-09 23:37:11

## Task Results
### Information Retrieval
- **Status:** success
- **Runtime:** 0.47s
| Metric | @1 | @3 | @5 | @10 |
|--------|----|----|----|----|
| NDCG   | 0.667 | 0.672 | 0.696 | 0.750 |
...
```

### 2. Test Data File

**Format**: JSON
**Location**: `data/test_data.json`
**Contains**:
- Test metadata (version, language, domain, description)
- Task configurations
- Actual test cases (queries, triplets, morphology pairs, etc.)
- Expected behaviors

**Purpose**: Provides context about what was tested, helps understand results

### 3. Optional: Previous Reports (for comparison)

**Format**: Markdown
**Location**: `output/previous_reports/`
**Purpose**: Track improvements over time, compare model versions

---

## Prompt Template

Use this prompt structure when asking an LLM to generate an evaluation report:

```markdown
## Context

I have evaluation results from the SemEval framework testing a semantic embedding model.

**Model Details**:
- Model Name: [e.g., "Turkish Financial Embeddings"]
- Domain: [e.g., "Finance", "General", "Medical"]
- Language: [e.g., "Turkish", "English", "Multilingual"]

**Test Suite Details**:
- Test Name: [from metadata.description]
- Total Tasks: [4 standard tasks]
- Runtime: [total execution time]

## Task

Generate a comprehensive evaluation report analyzing the model's performance.

## Required Inputs

**1. Evaluation Results**:
[Paste or reference the output/results.md or results.json file]

**2. Test Data Configuration**:
[Paste or reference data/test_data.json - at minimum the metadata section]

## Report Structure

### 1. Executive Summary
- Overall assessment (Excellent/Good/Fair/Poor)
- Production readiness verdict (Ready/Not Ready/Needs Work)
- Key findings (2-3 bullet points per task)
- Critical blockers (if any)
- Estimated timeline to production readiness

### 2. Task-by-Task Deep Dive

For each task, provide:

#### A. Core Metrics Table
- Present all metrics with scores
- Assessment labels (✅ Excellent, 🟢 Good, 🟡 Fair, 🔴 Poor)

#### B. Detailed Metric Interpretation

For EACH major metric:

**What it means**: Plain language explanation
- Define the metric in simple terms
- Explain what it measures

**Score Interpretation**:
- Translate numeric score to quality label
- Provide context (industry benchmarks if known)
- Explain what this score means practically

**Real-world examples**:
- Give concrete scenarios showing what this score enables/prevents
- Use actual test cases from the data when possible

**Context & Comparison**:
- Compare to typical ranges
- Note if score is above/below expectations

#### C. Performance Breakdowns

Analyze by:
- Difficulty level (if applicable)
- Category/Subcategory
- Any other relevant dimensions

For each breakdown:
- Present data in tables
- Identify patterns (where model excels/struggles)
- Provide hypotheses for why

#### D. Failure Analysis

For any failed or weak metrics:
- Show example test cases that failed
- Explain why they failed (root cause)
- Assess severity (Critical/High/Medium/Low)

#### E. Production Implications

- **Use cases where this performs well**: Specific applications
- **Use cases where this may struggle**: Limitations
- **Deployment considerations**: What to watch out for

### 3. Cross-Task Analysis

#### Strengths & Weaknesses Matrix
- List 5-7 key strengths with evidence
- List 5-7 key weaknesses with severity

#### Production Readiness Matrix
- Table showing each criterion, status, and whether it's a blocker

#### Pattern Recognition
- Cross-task correlations (e.g., "negation failure in robustness relates to analogy failures")
- Systemic issues (e.g., "bag-of-words behavior across tasks")

### 4. Recommendations

For each weakness, provide:

**Problem Statement**: Clear description of the issue

**Severity**: Critical/High/Medium/Low

**Proposed Solutions**:
1. Solution 1 (with pros/cons)
2. Solution 2 (with pros/cons)
3. Solution 3 (if applicable)

**Expected Improvement**: Quantified (e.g., "0% → 70% typo robustness")

**Effort Estimate**: Low/Medium/High (with rough time estimate)

**Priority**: 1-5 (1 = most urgent)

#### Action Plan

- **Phase 1**: Critical fixes (timeline)
  - List specific tasks
  - Milestones
- **Phase 2**: Quality improvements (timeline)
  - List specific tasks
  - Milestones
- **Phase 3**: Advanced features (timeline)
  - List specific tasks
  - Milestones

### 5. Conclusion

- Summarize overall assessment
- Restate production readiness
- Provide final recommendation (deploy/don't deploy/deploy with caveats)
- Estimated timeline to full production readiness

## Formatting Guidelines

### Tone & Style
- **Professional but accessible**: Technical accuracy without jargon overload
- **Actionable**: Every finding should connect to a recommendation
- **Evidence-based**: Support claims with data from results
- **Balanced**: Acknowledge both strengths and weaknesses

### Visual Elements
- Use emojis for quick visual scanning: ✅ 🟢 🟡 🔴 🚨 ⚠️ ❌
- Tables for comparative data
- **Bold** for emphasis on key findings
- Blockquotes for critical warnings

### Metric Interpretation Best Practices

For each metric, follow this pattern:

1. **Define**: What does this metric measure?
2. **Interpret**: What does this score mean?
3. **Contextualize**: How does it compare to benchmarks?
4. **Exemplify**: What real-world scenarios does this enable/prevent?
5. **Diagnose**: If poor, what's the root cause?

### Example Metric Interpretation

```markdown
#### NDCG@10: 0.750 🟢

**What it means**: The model achieves good ranking quality when considering the top 10 results.

**Interpretation**:
- Falls into the "Good" range (0.7-0.9)
- Relevant documents generally appear near the top
- Not perfect, but acceptable for most applications

**Example**: For the query *"Merkez Bankası faiz kararları..."*, the model likely ranks highly relevant documents about TCMB policy in the top 5-10 positions.

**Context**: Industry benchmarks for Turkish financial IR typically range 0.65-0.85. This score is above average.
```

### Root Cause Analysis Pattern

When analyzing failures, use this structure:

```markdown
#### Root Cause Analysis

**Observed Behavior**: [What we see in results]

**Evidence**: [Specific test cases showing the issue]

**Hypothesis**: [Why this is happening - technical explanation]

**Verification**: [How to confirm this hypothesis]

**Related Issues**: [Connection to other failures]
```

### Recommendation Pattern

```markdown
#### Priority 1: Fix [Problem Name] 🔴

**Problem**: [Concise problem statement with severity]

**Solutions**:

1. **[Solution Name]** (Recommended)
   - Approach: [How to implement]
   - Pros: [Advantages]
   - Cons: [Disadvantages]
   - Effort: [Low/Medium/High]

2. **[Alternative Solution]**
   - Approach: [How to implement]
   - Pros: [Advantages]
   - Cons: [Disadvantages]
   - Effort: [Low/Medium/High]

**Expected Improvement**: [Quantified before → after]

**Effort**: [Time estimate]

**Dependencies**: [What needs to be done first]
```

## Quality Checklist

Before finalizing the report, ensure:

### Completeness
- [ ] All 4 tasks analyzed in depth
- [ ] Every major metric interpreted
- [ ] All failures have root cause analysis
- [ ] Recommendations for all weaknesses
- [ ] Action plan with timelines

### Accuracy
- [ ] Metric interpretations are technically correct
- [ ] Score ranges and benchmarks are accurate
- [ ] Examples match actual test data
- [ ] Root cause hypotheses are plausible

### Actionability
- [ ] Each weakness has concrete recommendations
- [ ] Solutions include effort estimates
- [ ] Action plan has clear phases
- [ ] Success criteria are defined

### Clarity
- [ ] Non-experts can understand key findings
- [ ] Technical terms are explained
- [ ] Visual elements aid comprehension
- [ ] Report flows logically

### Professional Quality
- [ ] Consistent formatting throughout
- [ ] No typos or grammatical errors
- [ ] Tables are well-formatted
- [ ] Tone is balanced and objective

## Output Format

Save the report as: `EVALUATION_REPORT.md`

Include these sections:
1. Title with model name and date
2. Executive Summary (1-2 pages)
3. Detailed Task Analysis (10-15 pages)
4. Strengths & Weaknesses (2-3 pages)
5. Recommendations (3-5 pages)
6. Conclusion (1 page)

Total length: ~20-30 pages for comprehensive analysis

---

## Example Usage

### Full Prompt Example

```
I need you to generate a comprehensive evaluation report for a Turkish semantic embedding model.

**Model Details**:
- Model Name: Turkish Financial Embeddings
- Domain: Finance (with general Turkish coverage)
- Language: Turkish

**Evaluation Results**:
[See attached: output/results.md]

**Test Configuration**:
[See attached: data/test_data.json]

**Requirements**:
1. Analyze all 4 tasks in depth
2. For EACH metric, explain:
   - What it measures
   - What the score means
   - Real-world impact
   - If poor, what the root cause is
3. Identify production blockers (critical failures)
4. Provide actionable recommendations with effort estimates
5. Create a phased action plan with timelines

**Key Focus Areas**:
- Robustness issues are CRITICAL - emphasize if scores are low
- Production readiness assessment - be honest about deployment risks
- Concrete examples from test data to illustrate findings
- Quantified recommendations (e.g., "improve from X% to Y%")

**Output Format**:
- Markdown file: EVALUATION_REPORT.md
- 20-30 pages
- Professional tone but accessible
- Use emojis for visual scanning (✅🟢🟡🔴)
- Include all sections from the template

Generate the report following the structure in PROMPT_FOR_REPORT.md.
```

---

## Tips for Better Reports

### 1. Customize to Audience

**For Technical Teams**:
- Include architectural recommendations
- Deep dive into model internals
- Discuss training procedures

**For Business Stakeholders**:
- Focus on use cases and ROI
- Simplify technical explanations
- Emphasize business impact

**For Researchers**:
- Compare to academic benchmarks
- Discuss theoretical implications
- Suggest research directions

### 2. Use Data Effectively

- Don't just repeat numbers from results file
- Calculate derived metrics (gaps, improvements, correlations)
- Find patterns across tasks
- Connect quantitative findings to qualitative insights

### 3. Make It Actionable

Every finding should answer:
- What's the issue?
- Why does it matter?
- How do we fix it?
- What's the expected outcome?

### 4. Prioritize Ruthlessly

Not all issues are equal:
- Separate "production blockers" from "nice to haves"
- Focus recommendations on high-impact items
- Create clear priority tiers

### 5. Provide Context

- Explain what "good" looks like (benchmarks)
- Show industry standards
- Compare to similar models if known
- Acknowledge limitations of test suite

---

## Maintenance

### When to Update This Template

- After significant framework changes
- When new tasks are added
- Based on user feedback on report clarity
- When new metric types are introduced

### Versioning

Track changes to this template:
- **v1.0 (2025-01-09)**: Initial version
- **v1.1 (TBD)**: [Future updates]

---

## Related Documents

- [METRICS.md](METRICS.md) - Detailed metric explanations
- [USAGE.md](USAGE.md) - How to run evaluations
- [EVALUATION_REPORT.md](EVALUATION_REPORT.md) - Example report generated using this prompt

---

## Feedback

If using this template to generate reports, please note:
- What sections were most useful?
- What was unclear or missing?
- What additional guidance would help?

This helps improve the template for future use.

---

**Template Version**: 1.0
**Last Updated**: 2025-01-09
**Created for**: SemEval v0.1.0

**Made with ❤️ for reproducible, high-quality evaluation reports**
