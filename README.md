# INST447 Midterm Exam — Educational Text Complexity Analysis

> Course: INST447 — Data Sources and Manipulation  
> Section: 0101  
> Instructor: Wei Ai  
> Author: Myles J. Sartor  
> Date: November 23, 2025  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
   - [Source and Context](#source-and-context)
   - [Structure and Key Columns](#structure-and-key-columns)
   - [Annotation Scheme](#annotation-scheme)
3. [Requirements](#requirements)
4. [Installation and Setup](#installation-and-setup)
5. [How to Run](#how-to-run)
6. [Question-by-Question Summary](#question-by-question-summary)
   - [Q1: Dataset Structure, Missing Data, and Complexity Distributions](#q1-dataset-structure-missing-data-and-complexity-distributions)
   - [Q2: Decoding the Annotation Process via Regex](#q2-decoding-the-annotation-process-via-regex)
   - [Q3: Tier Classification Consistency](#q3-tier-classification-consistency)
   - [Q4: Predicting Vocabulary Complexity from Tier 3 Word Counts](#q4-predicting-vocabulary-complexity-from-tier-3-word-counts)
   - [Q5: LLM Annotation Comparison](#q5-llm-annotation-comparison)
   - [Q6: Cross-Grade Rating Sanity Check](#q6-cross-grade-rating-sanity-check)
   - [Q7: AI Assistant Reflection](#q7-ai-assistant-reflection)
   - [Q8: The Flesch-Kincaid Mystery (Bonus)](#q8-the-flesch-kincaid-mystery-bonus)
7. [Key Findings](#key-findings)
8. [Limitations](#limitations)
9. [Visualizations Summary](#visualizations-summary)

---

## Project Overview

This exam-based analysis conducts a guided exploratory data analysis of a multi-annotator educational text complexity dataset assembled by the CZI Learning Commons as part of their Literacy Evaluators project. The dataset contains over 1,000 educational texts annotated for sentence and vocabulary complexity by teams of expert educators evaluating texts intended for Grade 3 and Grade 4 students.

The analysis spans eight questions covering dataset profiling, annotation process decoding, consistency analysis of vocabulary tier classifications, predictive relationships between Tier 3 word counts and complexity scores, a comparison of human annotation against LLM-generated annotation, a cross-grade sanity check on rating logic, an AI tool usage reflection, and a bonus investigation into a discrepancy in Flesch-Kincaid readability scores.

---

## Dataset

### Source and Context

**File:** `text_complexity.csv`  
**Origin:** CZI Learning Commons — Literacy Evaluators Project  
**Coverage:** Over 1,000 educational texts evaluated for Grades 3 and 4  
**Annotation method:** Multiple annotators per text; scores consolidated using the Dawid-Skene statistical method for handling annotator disagreement

The dataset was created to help educators select reading materials calibrated to the appropriate developmental level for elementary students. Each text was independently assessed by multiple human annotators who rated sentence complexity, vocabulary complexity, and identified Tier 2 and Tier 3 vocabulary words.

---

### Structure and Key Columns

| Column                      | Type   | Description                                                                         |
|-----------------------------|--------|-------------------------------------------------------------------------------------|
| `Clear ID`                  | string | Unique identifier for each text. Many texts appear twice — once per grade level.    |
| `Grade`                     | int    | Grade level the text was evaluated for (3 or 4)                                     |
| `Sentence Score`            | string | Consolidated sentence complexity rating (see categories below)                      |
| `Vocabulary Score`          | string | Consolidated vocabulary complexity rating (see categories below)                    |
| `Tier 2 Words`              | string | Comma-separated list of general academic vocabulary words identified in the text    |
| `Tier 3 Words`              | string | Comma-separated list of domain-specific terminology identified in the text           |
| `Sentence Score Rationale`  | string | Free-text rationale from each sentence annotator, prefixed by annotator ID          |
| `Vocabulary Score Rationale`| string | Free-text rationale from each vocabulary annotator, prefixed by annotator ID        |
| `Flesch Kincaid`            | float  | Flesch-Kincaid readability grade-level score computed from sentence length/syllables|
| `archaic` / `complex`       | varies | Word-level annotation fields with high rates of missingness (20%+)                  |

---

### Annotation Scheme

Complexity scores use five categories:

| Category             | Numeric mapping (used in Q6) |
|----------------------|------------------------------|
| `not scored`         | -1                           |
| `slightly complex`   | 1                            |
| `moderately complex` | 2                            |
| `very complex`       | 3                            |
| `exceedingly complex`| 4                            |

Vocabulary tier definitions used throughout the analysis:

- **Tier 2 Words:** General academic vocabulary appearing across many subject domains — words that mature language users would recognize but that elementary students may not (e.g., "analyze", "contribute", "establish").
- **Tier 3 Words:** Domain-specific terminology unique to a particular field or subject — words rarely encountered outside that domain (e.g., "photosynthesis", "denominator", "peninsula").

---

## Requirements

- Python 3.x (3.7+ recommended)
- The dataset file `text_complexity.csv` placed in the same directory as the notebook

Python packages:

```bash
pip install pandas numpy matplotlib seaborn
```

| Package       | Purpose                                                         |
|---------------|-----------------------------------------------------------------|
| `pandas`      | All data loading, cleaning, grouping, pivoting, and filtering   |
| `numpy`       | Numerical operations and array handling                         |
| `matplotlib`  | Bar charts, pie charts, and scatter plots                       |
| `seaborn`     | Styled scatter plots for cross-grade comparison (Q6)            |
| `re`          | Regular expression parsing of annotator rationale fields (Q2)   |
| `collections` | `Counter` and `defaultdict` for word-level aggregation          |
| `typing`      | Type hints for custom analysis functions                        |

---

## Installation and Setup

1. Download or clone the project repository.
2. Place `text_complexity.csv` in the same directory as `Midterm_Exam.ipynb`.
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Run all cells from top to bottom. No API keys or external network access are required.

---

## How to Run

Open `Midterm_Exam.ipynb` and execute all cells sequentially. The notebook is divided into clearly labeled sections, one per exam question. Each question section is self-contained except for the shared `df` DataFrame initialized in Q1 and the `parse_word_list()` helper function defined in Q3, which is reused in Q4. The LLM annotation data in Q5 is hardcoded directly into the notebook — no live API calls are made.

---

## Question-by-Question Summary

### Q1: Dataset Structure, Missing Data, and Complexity Distributions

**Goal:** Establish foundational understanding of what the dataset contains and where it is incomplete before any further analysis.

**Approach:**
- Load `text_complexity.csv` with `pd.read_csv()`
- Print dataset shape (rows x columns) and column data types
- Count unique `Clear ID` values and compare to total row count — more rows than unique IDs indicates the same texts appear multiple times, once per grade level
- Calculate per-column null counts and percentages, sort descending
- Produce pie charts and bar charts for both `Sentence Score` and `Vocabulary Score` distributions, including the `"not scored"` category as a valid category

**Key results:**
- Two columns (`archaic` and `complex`) exceed 20% missingness, reflecting the difficulty of consistently annotating these specialized features across all texts
- The remaining columns have negligible missingness (1% or less)
- Sentence Score distribution: 36.2% very complex, 30.1% moderately complex, 15.2% slightly complex, 8.6% not scored, 7.9% exceedingly complex
- Vocabulary Score distribution: 46.9% not scored — a dramatically higher unscored rate than for sentence scoring, likely reflecting annotators' difficulty classifying vocabulary for many texts

---

### Q2: Decoding the Annotation Process via Regex

**Goal:** Determine how many annotators evaluated each text, since this information is embedded in the free-text rationale fields rather than stored as an explicit column.

**Approach:**
- Write a function `count_annotators(rationale_text, prefix)` that applies `re.findall()` with the pattern `{prefix}_annotator_(\d+):` to extract all annotator identifier numbers from a rationale string
- Apply `len(set(matches))` to count unique annotators per text, guarding against duplicate mentions of the same annotator ID
- Apply the function separately to `Sentence Score Rationale` (prefix: `ss`) and `Vocabulary Score Rationale` (prefix: `vocab`)
- Produce side-by-side bar charts of the resulting annotator count distributions

```python
def count_annotators(rationale_text, prefix):
    if pd.isna(rationale_text):
        return 0
    pattern = rf'{prefix}_annotator_(\d+):'
    matches = re.findall(pattern, rationale_text)
    return len(set(matches))
```

**Key results:**
- Most texts were evaluated by exactly 3 sentence annotators
- Vocabulary annotator counts were most frequently 0 or 3, with a similar but slightly more variable distribution than sentence annotators
- The similar patterns suggest the same annotation teams tended to assess both dimensions of a given text, supporting a coordinated evaluation methodology

---

### Q3: Tier Classification Consistency

**Goal:** Test whether vocabulary tier labels are applied consistently across annotators and across grade levels, or whether the distinction between Tier 2 and Tier 3 is subjective and context-dependent.

**Part A — Within-Grade Consistency (Grade 3):**
- Filter to `Grade == 3` texts only
- Define `parse_word_list()` to split comma-separated word strings into lowercase lists, handling `NaN` and `"not scored"` entries
- Build set unions of all Tier 2 and Tier 3 words across all Grade 3 texts
- Take the set intersection to find words appearing in both tiers within the same grade level

```python
def parse_word_list(word_string):
    if pd.isna(word_string) or word_string == 'not scored':
        return []
    return [word.strip().lower() for word in word_string.split(',')]
```

**Result:** 767 words appeared in both Tier 2 and Tier 3 across different Grade 3 texts. Example words with inconsistent classification: "relations", "asteroids", "gas". This confirms that the Tier 2 / Tier 3 boundary is not objective — different annotators classify the same word differently depending on their individual interpretation of domain-specificity.

**Part B — Cross-Grade Classification:**
- Repeat the set union procedure for Grade 4 texts
- Find words classified as Tier 3 in Grade 3 texts but Tier 2 in Grade 4 texts (and vice versa)

**Result:** 343 words were Tier 3 in Grade 3 but Tier 2 in Grade 4 (e.g., "poet", "gas", "warlike"). 306 words were Tier 2 in Grade 3 but Tier 3 in Grade 4 (e.g., "distant", "glaciers", "cells"). The tier-switching pattern demonstrates that tier classification is context-dependent rather than intrinsic to the word — what is domain-specific for 3rd graders may be general academic vocabulary by 4th grade.

---

### Q4: Predicting Vocabulary Complexity from Tier 3 Word Counts

**Goal:** Test the hypothesis that texts with more domain-specific vocabulary (more Tier 3 words) will be rated as more complex.

**Approach:**
- Count the length of each parsed Tier 3 word list per text using `parse_word_list()` from Q3
- Filter out rows where `Vocabulary Score == "not scored"`
- Group by `Vocabulary Score` and compute the mean Tier 3 word count for each complexity level
- Produce a bar chart with complexity level on the x-axis and mean Tier 3 count on the y-axis, and a companion box plot showing the distribution within each level

**Key results:**
- A clear positive progression exists: mean Tier 3 count increases monotonically from "slightly complex" through "exceedingly complex"
- Box plots reveal substantial within-category variation, meaning Tier 3 count alone is not a perfect predictor — other factors such as sentence structure, concept density, and vocabulary frequency also influence complexity scores
- Despite the annotation inconsistencies documented in Q3, the overall relationship between Tier 3 word density and complexity rating is robust, validating the conceptual framework underlying the annotation scheme

---

### Q5: LLM Annotation Comparison

**Goal:** Evaluate whether a Large Language Model can replicate human annotator decisions for Tier 2 and Tier 3 vocabulary classification across five educational texts.

**Prompt Design:**
A structured prompt was constructed incorporating the official Tier 2 and Tier 3 definitions from the Learning Commons documentation. The prompt instructed the model to:
1. Identify Tier 2 words (general academic vocabulary appearing across domains)
2. Identify Tier 3 words (domain-specific terminology for the given grade level)
3. Return results as a JSON object with keys `"tier2_words"` and `"tier3_words"`

The five test texts covered: the American Revolution, human biology (blood composition), geological formation (the Grand Canyon), climate science, and electricity/physics.

**Overlap Calculation:**
For each text, the intersection between human-annotated word sets and LLM-annotated word sets was computed for both tiers. Overall overlap rates across all five texts:

| Tier      | Overlap               |
|-----------|-----------------------|
| Tier 3    | 68.2% (58 / 85 words) |
| Tier 2    | 49.4% (42 / 85 words) |

**Key results:**
- Tier 3 overlap was substantially higher because domain-specific terms (e.g., "Battle of Saratoga", "hemoglobin", "coulomb") are unambiguous — the LLM and humans converge on the same clearly specialized vocabulary
- Tier 2 overlap was lower because the boundary between "general academic vocabulary" and "common language" is subjective — the LLM interprets Tier 2 more liberally and identifies 20–30% more words overall
- Cross-tier mismatches (LLM classifying a human Tier 2 word as Tier 3, or vice versa) were most common in texts dealing with science topics where terminology occupies a middle ground
- LLM performance varied by subject domain, performing best on well-defined scientific fields and worst on social studies and humanities topics where the distinction between general and domain-specific vocabulary is less clear

---

### Q6: Cross-Grade Rating Sanity Check

**Goal:** Verify that complexity ratings follow a logical educational pattern — if the same text is evaluated for both Grade 3 and Grade 4, the Grade 3 rating should be equal to or higher than the Grade 4 rating (the same material is more challenging for younger students).

**Approach:**
- Filter to texts with more than one row in the dataset (i.e., texts present for both grades)
- Pivot the data so each row has Grade 3 and Grade 4 complexity scores in separate columns
- Convert string complexity labels to numeric values using a mapping dictionary (`slightly=1, moderately=2, very=3, exceedingly=4, not scored=-1`)
- Compute the difference (Grade 4 score minus Grade 3 score) for both Sentence Score and Vocabulary Score
- Count the distribution of differences and flag unexpected cases (negative difference = text rated more complex for Grade 4 than Grade 3)
- Produce scatter plots of Grade 3 vs Grade 4 scores with a diagonal reference line indicating perfect agreement

```python
def complexity_to_numeric(score):
    mapping = {
        'slightly complex': 1, 'moderately complex': 2,
        'very complex': 3, 'exceedingly complex': 4, 'not scored': -1
    }
    return mapping.get(score, -1)
```

**Key results:**
- Out of 491 rated text pairs, the majority showed expected patterns (same complexity rating or higher for Grade 4)
- 379 texts showed reversed patterns (rated more complex for Grade 3 than Grade 4). Examination of examples revealed that these often involved exceedingly or very complex sentence scores for Grade 3 paired with lower or unscored assessments for Grade 4, suggesting annotation inconsistency between different rating teams rather than a genuine reversal in educational difficulty
- Scatter plots showed most data points at or above the diagonal line for Sentence Score; Vocabulary Score showed more dispersion, consistent with the higher not-scored rate found in Q1

---

### Q7: AI Assistant Reflection

AI was used selectively across several questions for specific technical tasks. It was most valuable for:

- Generating and refining regular expression patterns for extracting annotator counts from free-text rationale fields (Q2)
- Suggesting pandas operations for multi-step transformations such as pivot tables, set intersections, and cross-grade comparison logic (Q6)
- Debugging output formatting for complex data structures

AI was verified by testing all generated code on small subsets of the data before applying it to the full dataset. Detailed prompts specifying data structures, column names, and expected outputs were essential to obtaining useful responses — vague prompts produced generic code that required significant modification. AI was not used for writing interpretations or analytical conclusions; all domain-level reasoning about educational complexity, annotator behavior, and grade-level expectations was developed independently.

---

### Q8: The Flesch-Kincaid Mystery (Bonus)

**Goal:** Investigate why the mean Flesch-Kincaid score across the dataset is approximately 7.2 — suggesting text appropriate for 7th graders — when the texts are explicitly designed and annotated for 3rd and 4th graders.

**Code approach:**
- Extract the `Flesch Kincaid` column using `pd.to_numeric()` with `errors="coerce"` to handle any string values
- Compute the column mean
- Segment texts by Vocabulary Score complexity and compare mean FK scores between high-complexity and low-complexity groups

**Finding:** Flesch-Kincaid relies solely on sentence length (average words per sentence) and syllable count (average syllables per word). It entirely ignores vocabulary sophistication and conceptual density. Educational texts for early elementary students are frequently written in short, syntactically simple sentences while simultaneously introducing highly specialized multi-syllable domain vocabulary (e.g., "photosynthesis", "hemoglobin"). The Flesch-Kincaid formula interprets those technical terms as evidence of difficulty appropriate for an older reader, even though the surrounding sentence structure is designed for young students. The formula was developed for general readability assessment and lacks any mechanism for accounting for curriculum context, developmental appropriateness, or the distinction between technical vocabulary and sentence complexity. As a result, it systematically overestimates the grade level of educational science and social studies texts written for elementary students.

---

## Key Findings

- The dataset has over 20% missingness in two annotation-specific columns (`archaic`, `complex`), reflecting the practical limits of labor-intensive content annotation
- Most texts were evaluated by 3 sentence annotators and 0 or 3 vocabulary annotators, indicating a structured but not perfectly uniform annotation process
- 767 words within Grade 3 texts alone appeared in both Tier 2 and Tier 3 categories, and hundreds more switched tiers between Grade 3 and Grade 4. Tier classification is substantially subjective and context-dependent rather than an intrinsic property of the word
- Tier 3 word count is a consistent positive predictor of Vocabulary Score complexity despite annotation inconsistencies, validating the underlying conceptual framework
- LLMs achieve 68.2% overlap with humans on Tier 3 annotation and 49.4% on Tier 2. LLMs systematically identify more vocabulary items overall and diverge most on words at the boundary between general academic and domain-specific categories
- Cross-grade complexity ratings are broadly logical but include 379 unexpected reversals, most attributable to annotation inconsistency between different rater teams
- Flesch-Kincaid scores averaging 7.2 for texts labeled as Grade 3–4 content highlight a fundamental limitation of automated readability formulas when applied to educational materials

---

## Limitations

- The dataset is static and reflects the judgment of a specific set of educator-annotators working within a particular annotation framework. Different annotation teams or updated guidelines could yield substantially different tier classifications
- The OMDb and IMDb notes do not apply here, but the LLM comparison in Q5 was performed on only 5 texts. A larger sample would produce more reliable aggregate overlap statistics
- The cross-grade sanity check in Q6 uses `aggfunc='first'` in the pivot table for cases where a given `Clear ID` has more than one row per grade. This may not always select the most representative or consolidated annotation for that text
- Numeric mapping of complexity categories (slightly=1 through exceedingly=4) treats the scale as interval rather than ordinal. The actual psychological distance between complexity categories is unknown and may not be uniform
- Regex-based annotator counting in Q2 assumes a consistent naming convention (`ss_annotator_N:` and `vocab_annotator_N:`) throughout all rationale fields. Any deviation in formatting would cause undercount

---

## Visualizations Summary

| Question | Visualization                                      | Type               | Key Variables                                 |
|----------|----------------------------------------------------|--------------------|-----------------------------------------------|
| Q1       | Sentence Score Distribution                        | Pie chart          | Sentence Score categories                     |
| Q1       | Vocabulary Score Distribution                      | Pie chart          | Vocabulary Score categories                   |
| Q1       | Sentence Score Distribution                        | Bar chart (skyblue)| Sentence Score categories vs count            |
| Q1       | Vocabulary Score Distribution                      | Bar chart (coral)  | Vocabulary Score categories vs count          |
| Q2       | Distribution of Sentence Annotators per Text       | Bar chart (red)    | Annotator count vs number of texts            |
| Q2       | Distribution of Vocabulary Annotators per Text     | Bar chart (green)  | Annotator count vs number of texts            |
| Q4       | Mean Tier 3 Word Count by Vocabulary Complexity    | Bar chart (pink)   | Complexity level vs mean Tier 3 count         |
| Q4       | Distribution of Tier 3 Word Count by Complexity   | Box plot           | Complexity level vs Tier 3 count distribution |
| Q5       | Tier 2 vs Tier 3 Overlap Percentage by Text        | Grouped bar chart  | Text ID vs overlap percentage per tier        |
| Q5       | Distribution of Human-only vs LLM-only Words       | Box plot           | T2/T3 Human-only vs LLM-only word counts      |
| Q5       | Cross-Tier Classification Mismatches               | Bar chart (orange) | Text ID vs number of cross-tier mismatches    |
| Q5       | Total Vocabulary Items Identified                  | Grouped bar chart  | Human total vs LLM total words per text       |
| Q6       | Sentence Score: Grade 3 vs Grade 4                 | Scatter plot       | Grade 3 complexity vs Grade 4 complexity      |
| Q6       | Vocabulary Score: Grade 3 vs Grade 4               | Scatter plot       | Grade 3 complexity vs Grade 4 complexity      |
