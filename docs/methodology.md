# Methodology Notes

## Study Design

The project combines a review-text analytics workflow with a questionnaire-based consumer-behavior study. The behavioral part compares responses under real-review and fake-review conditions and models how review type affects booking intention through perceived authenticity, trust, and perceived risk.

## Data Cleaning

The final questionnaire output applies two screening rules:

- adult respondents are retained;
- respondents with relevant online shopping or hotel-booking experience are retained.

After screening, the final sample contains 68 valid observations.

## DID-Style Comparison

The DID-style component compares before/after booking intention changes across real-review and fake-review conditions. It is useful as an intuitive contrast, but the estimate is reported cautiously because the questionnaire sample is small.

## PLS-SEM-Style Path Model

The SEM component estimates the following mechanism:

```text
Review Type -> Review Authenticity -> Trust -> Booking Intention
                         |              |
                         v              v
                  Perceived Risk ----> Booking Intention
```

Key path results include:

- Review Type -> Review Authenticity: beta = 0.485, p = 6.297e-10
- Review Authenticity -> Trust: beta = 0.749, p = 0
- Trust -> Booking Intention: beta = 0.457, p = 2.633e-05
- Booking Intention R2 = 0.667

## Measurement Quality

The model checks reliability and validity through Cronbach's alpha, composite reliability, AVE, HTMT, Fornell-Larcker, cross-loadings, and inner VIF. The curated CSV files in `results/` contain the detailed values.

## NLP Pipeline

The fake-review detection workflow is structured around:

1. text normalization and cleaning;
2. optional stop-word removal, stemming, or lemmatization;
3. feature extraction with TF-IDF and simple linguistic features;
4. supervised classification using scikit-learn;
5. evaluation using accuracy, precision, recall, F1, and confusion matrix outputs.

## Limitations

The questionnaire component is best interpreted as pilot evidence because the sample size is limited. The repository therefore presents results as a pilot empirical research project, emphasizing clean workflow, transparent assumptions, and reproducible outputs.
