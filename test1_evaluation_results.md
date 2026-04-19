# Headline Persuasion Classifier Evaluation Report
**Dataset**: `Test1.json` (Real-world, out-of-distribution test set)
**Models Compared**: "New Normal" trained model vs. Hierarchical trained model

## 1. Hierarchical Pipeline Evaluation

**Overall Results:**
- **Overall Route Accuracy:** 71.55% (0.7155)
- **Macro F1:** 0.7066
- **Topic Accuracy (Stage 1):** 94.83% (0.9483)
- **Topic Misclassifications:** 6 / 116

**Confusion Matrix (Route):**
```text
                  Precision    Recall  F1-Score   Support

   central_route     0.8500    0.5667    0.6800        30
peripheral_route     0.5510    0.8710    0.6750        31
   neutral_route     0.8298    0.7091    0.7647        55
```

*Notes: The hierarchical pipeline leans toward predicting `peripheral_route` when in doubt, but its `central_route` precision is quite high—when it says "central", it is usually correct (85% of the time).*

---

## 2. "New Normal" Evaluation

**Overall Results:**
- **Overall Accuracy:** 66.38% (0.6638)
- **Macro F1:** 0.6481

**Confusion Matrix:**
```text
                  Precision    Recall  F1-Score   Support

   central_route     0.7368    0.4667    0.5714        30
peripheral_route     0.5000    0.9032    0.6437        31
   neutral_route     0.8537    0.6364    0.7292        55
```

---

## Key Takeaways

1. **Architecture Superiority:** The hierarchical approach is a clear winner on this harder dataset, achieving an accuracy of ~71.55% vs the normal model's ~66.38% (a +5% raw improvement). Breaking the logic down into topic-first significantly improves generalization.
2. **Central Route Classification:** Both models struggle with overall recall for `central_route`, frequently confusing it. However, the hierarchical model does so much more accurately (0.85 precision compared to the normal model's 0.73).
3. **Overpredicting Peripheral:** Both models exhibit low precision for `peripheral_route` (≈0.50–0.55) alongside high recall. This indicates they often flag central or neutral headlines as peripheral incorrectly, perhaps due to oversensitivity to sensationalistic keywords or "clickbaity" language.
