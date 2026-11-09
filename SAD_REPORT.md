# SAD Mini-Activity and Reflection

## SAD mini-analysis

| Element | Student response |
|---|---|
| Problem statement | Academic risk may only be noticed after poor results. This prototype explores whether available student indicators can help flag learners who may need additional academic support. |
| Users | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators such as study time, absences, failures, G1 and G2. G3 is the target and is not an input. |
| Process | Validate data → preprocess → apply trained model → classify risk. |
| Outputs | AT RISK / NOT AT RISK label and model evaluation summary. |
| Key limitation | A model trained using one Portuguese secondary-school dataset may not generalize to another institution or population. |

**Mini use case:** User uploads CSV → system checks required columns → model predicts risk → system displays and exports results.

## Reflection answers

**1. What is the prediction target, and why is G3 excluded from the input features?**

The target is final Mathematics performance risk. If G3 is below 10, the label is AT RISK; if G3 is 10 or higher, the label is NOT AT RISK. G3 is excluded because it is used to create the target. Including it as an input would reveal the answer to the model and cause target leakage.

**2. What does the model's recall for the 'At risk' class tell you?**

It tells us what proportion of all actually at-risk students were correctly identified. Low recall means the model misses more at-risk students, so some learners who may need support are not flagged.

**3. Why might a model trained on this dataset perform differently on students from your institution?**

The dataset comes from one Portuguese secondary-school setting. Our institution may have different students, grading systems, teaching methods, attendance patterns, and support services. These differences can affect model performance, so local validation is needed.

**4. What action should a faculty member take after seeing an 'AT RISK' prediction?**

The faculty member should verify the student's records, review their academic situation, and speak with the student to identify possible needs. The prediction should guide supportive review, not be treated as a final judgment or automatic decision.

## Responsible-use note
Predictions are not diagnoses and must not be used as the sole basis for real student decisions.
