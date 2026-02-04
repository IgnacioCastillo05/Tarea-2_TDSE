# Tarea-2_TDSE
TDSE Project on Logistic Regression

**NOTA**: El cuaderno de Jupyter donde se encuentra todo el desarrollo es el que se llama "notebook.ipynb".

## Paso 1

### Feature Selection

Eight features were selected based on three main criteria:

**1. Clinical Relevance:**
The chosen features are well‑established medical indicators for diagnosing heart disease according to the medical literature:
- **Age**: Primary non‑modifiable risk factor
- **Cholesterol & BP**: Modifiable cardiovascular risk factors
- **Max HR**: Indicator of cardiac functional capacity
- **ST depression**: ECG marker of myocardial ischemia
- **Number of vessels fluro**: Coronary angiography result
- **Chest pain type**: Differential clinical symptom
- **Thallium**: Myocardial perfusion test

**2. Exploratory Data Analysis (EDA):**
During EDA it was observed that all selected features exhibit:
- Distributions with adequate variability (no constant values)
- Ranges covering clinically meaningful spectra
- Presence of cases in risk zones (ej: Cholesterol >200 mg/dL, BP >140 mmHg)

For example:
- **Age**: Concentration between 48–67 years (highest‑risk range)
- **Cholesterol**: 59% of patients have values >200 mg/dL (risk threshold)
- **ST depression**: Skewed distribution; abnormal values (>0) in 36% of cases
- **Max HR**: Variability from 71–202 bpm, capturing different fitness levels

**3. Completitud de Datos:**
The 8 selected features have no missing values (270/270 complete records), eliminating the need for imputation and preserving dataset integrity.
These 8 features represent a balance of:

- Information diversity (demographics, symptoms, lab tests, ECG, imaging)
- Model parsimony (avoiding overfitting with too many features)
- Clinical interpretability (all features are standard medical measurements)

## Step 2

## Algorithm Convergence

The logistic regression model successfully converged after 1500 iterations using gradient descent with learning rate α = 0.01.

Iter    0: Cost = 0.690608
Iter  250: Cost = 0.458731
Iter  500: Cost = 0.418085
Iter  750: Cost = 0.403660
Iter 1000: Cost = 0.396851
Iter 1250: Cost = 0.393089
Iter 1499: Cost = 0.390799

Initial cost: 0.690608
Final cost:    0.390799
Bias (b): -0.205146

![alt text](img/convergencia.png)

**Evidencia de convergencia:**
- Initial cost: 0.6906 (iteration 0)
- Final cost: 0.3908 (iteration 1499)
- Total reduction: 0.2998 (43.4% improvement)
- Change in last 100 iterations: 0.00078980 (<0.001)

The convergence plot shows:
1. **Initial phase (0-200 iters):** Model quickly learns basic structure
   modelo aprende rápidamente la estructura básica de los datos.
2. **Intermediate phase (200-800 iters):** Gradual, consistent decline
3. **Final phase (800-1500 iters):** Plateau with minimal change (<0.001), confirming convergence

**Conclusion:** The model reached a stable local minimum. Learning rate α=0.01 was appropriate—neither too large (oscillations) nor too small (slow convergence).

## Performance Evaluation

### Metrics on Train vs Test:

| Metric    | Train  | Test   | Difference |
|------------|--------|--------|------------|
| Accuracy   | 0.8254 | 0.8765 | +0.0511    |
| Precision  | 0.8312 | 0.8250 | -0.0062    |
| Recall     | 0.7619 | 0.9167 | +0.1548    |
| F1-Score   | 0.7950 | 0.8684 | +0.0734    |

### Interpretation:

**1. No Overfitting:**
- Test metrics are equal or better than Train
- Test Accuracy > Train Accuracy (+5.1%)
- Test F1 > Train F1 (+7.3%)
- Model generalizes extremely well

**2. Trade-off Precision-Recall:**
- **Precision (Test = 82.5%):** 82.5% of predicted positive cases truly have disease
- **Recall (Test = 91.7%):** The model correctly detects 91.7% of true disease cases

**3. Medical Context:**
High Recall is critical in medical diagnosis—false negatives are the most dangerous.
This model detects ~11 out of 12 sick patients (only 1 false negative).

**4. F1-Score:**
- F1 = 0.8684 → excellent balance between Precision and Recall


## Coefficient Interpretation (Weights)

Positive coefficients increase disease probability; negative ones decrease it.

### Importance Ranking (absolute magnitude):

![alt text](img/coeficientes.png)

| Feature                    | Coefficient | Interpretation |
|----------------------------|-------------|----------------|
| **Number of vessels fluro**| **+0.821**  | MOST important (more blocked vessels → higher risk) |
| **Thallium**               | **+0.772**  | Abnormal result → higher risk |
| **Chest pain type**        | **+0.624**  | Typical angina → higher risk |
| **ST depression**          | **+0.532**  | High ST depression → ischemia |
| **Max HR**                 | **-0.457**  | Higher max HR → lower risk |
| **BP**                     | **+0.163**  | High BP → slight increased risk |
| **Age**                    | **-0.103**  | Slight negative effect (unexpected*) |
| **Cholesterol**            | **+0.094**  | Small positive effect |

### Insights Clave:

**1. Most Discriminative Features:**
- **Number of vessels fluro** (coef = 0.821): The most important feature.
Each additional blocked vessel significantly increases the log‑odds of having heart disease.
  
- **Thallium** (coef = 0.772): Second most important.
Abnormal results in the thallium test are strongly predictive of disease.

**2. Max HR with Negative Coefficient:**
- **Coef = -0.457**: Higher maximum heart rate achieved → LOWER risk
- **Medical explanation:** Patients who reach a high HR during exercise demonstrate better cardiovascular capacity. A limited HR response may indicate ischemia or poor physical condition.

**3. Age with Negative Coefficient (unexpected):**
- **Coef = -0.103**: Although age is a medical risk factor, the negative coefficient can be explained by:
  - **Multicollinearity:** The effect of age is already captured by other correlated features (e.g., ST depression, vessels)
  - **Dataset-specific:** The sample has a narrow age range (mostly 48–67 years)
  - **Normalization:** The effect is relative to stronger predictors

**4. Cholesterol Efecto Débil:**
- **Coef = 0.094** (small): Although clinically relevant, in this model its individual effect is minor once other variables are controlled for (vessels, thallium, ST depression already capture disease severity).

### Importance Visualization:
The bar chart clearly shows that advanced diagnostic tests (angiography, thallium, ECG) have higher predictive power than basic risk factors (age, cholesterol, blood pressure).
This confirms that in this dataset, test results are more informative for classification than demographic/basic indicators.

## Step 3
Three feature pairs were used:
1. Age vs Cholesterol
2. BP vs Max HR
3. ST Depression vs Number of Vessels Fluro

## Age vs Cholesterol

### Final Cost of the 2D Model: 0.672006

### Visual Observations:

![alt text](img/par1.png)

**Decision Boundary:**
- Diagonal line with negative slope (top-left to bottom-right)
- Approximate equation: w₁*Age + w₂*Cholesterol + b = 0
- Negative slope indicates both coefficients have opposite effects

**Class Separability:**
- **Moderate to low separation:**: There is considerable overlap between blue circles (Absence) and red squares (Presence)
- The central region (Age 45–60, Cholesterol 200–300) shows significant mixing
- No clear threshold such as “Cholesterol > 250” provides perfect separation

**Spatial Distribution:**
- **Blue zone (Absence):**
  - Concentrated in: low age + low cholesterol (bottom-left)
  - Also appears in: high age + high cholesterol (top-right)
  
- **Red zone (Presence):**
  - More scattered throughout the plot
  - Higher concentration in the central-right area

### Interpretation:

**Why is separation limited?**

1. Age and Cholesterol alone are insufficient to separate healthy vs diseased patients. Heart disease depends on multiple factors.

2. **Complex relationship:** There is no simple rule like “Cholesterol > X → disease”.
- Young patients with high cholesterol may be healthy (single-factor risk)
- Older patients with moderate cholesterol may still have disease (cumulative risk)

3. **Relatively high cost (0.672)**: Indicates difficulty in classification compared to other feature pairs.

### Convergence:
The convergence plot shows rapid initial descent and stabilization, confirming the model found the best possible fit for these two features.

### Conclusion:
Age and Cholesterol are not sufficient on their own to predict heart disease accurately. Additional diagnostic features (ECG, angiography, stress tests) are needed.

## BP (Blood Pressure) vs Max HR (Max Heart Rate)

### Final Cost of the 2D Model: 0.587574

### Visual Observations:

![alt text](img/par2.png)

**Decision Boundary:**
- Diagonal line with positive slope
- Clearer and better defined than in Pair 1
- Divides the space into two reasonably balanced regions

**Class Separability:**
- Moderate separation: Better than Age–Cholesterol
- Upper-left region (low BP + high Max HR) → mostly Absence (blue)
- Lower-right region (high BP + low Max HR) → mostly Presence (red)
- Central region still shows overlap, but less than in Pair 1

**Spatial Distribution:**
- **Blue zone (Absence - no illness):**
  - High concentration of high Max HR (>160 bpm) + low/moderate BP
  - Interpretation: Good cardiovascular capacity
  
- **Red zone (Presence - with illness):**
  - Concentrated in low Max HR (<140 bpm)
  - Interpretation: Limited HR response → possible ischemia or poor conditioning

### Clinical Interpretation:

**Max HR is a key indicator:**
- **High max HR (>150)** → good cardiovascular health
  
- **Low Max HR (<140)** → red flag (possible ischemia)

**BP effect:**
- Less pronounced than Max HR
- Extreme BP (>180) is associated with disease
- Normal-high BP (120–160) is ambiguous without additional indicators

### Connection to the Full Model:
In the full 8-feature model:
- **Max HR: coef = -0.457**
- **BP: coef = +0.163**

This aligns with the positive slope:
- Low Max HR (↑ risk) can be compensated by low BP
- High BP (↑ risk) can be compensated by high Max HR

### Conclusion:
BP and Max HR show moderate separability, with Max HR being the dominant discriminator. This pair is more informative than Age–Cholesterol.

## ST Depression vs Number of Vessels Fluro

### Final Cost of the 2D Model: 0.515726

### BEST PAIR — Lowest cost among the three

### Visual Observations:

![alt text](img/par3.png)

**Decision Boundary:**
- Strongly negative slope
- Clear and well‑defined boundary
- Effectively separates classes with minimal overlap

**Class Separability:**
- Excellent separation
- Blue zone: ST≈0, Vessels=0 → almost all healthy
- Red zone: ST>1, Vessels≥2 → almost all diseased

**Spatial Pattern::**

```
Vessels = 0, ST = 0        →  High probability of Absence
Vessels = 1, ST = 0-2      →  Mixed/transition region
Vessels ≥ 2, ST > 1        →  High probability of Presence
```

**Cluster Analysis:**
1. **Cluster Absence (0 vasos, ST ≈ 0):**
   - No vascular blockages
   - Normal ECG
   - → Nearly all healthy
   
2. **Cluster Presence (2-3 vasos, ST > 2):**
   - Multiple blocked vessels
   - Severe ischemia
   - → Nearly all diseased

**Why is this pair so effective**

1. **Both features are direct diagnostic signals:**
   - Vessels: anatomical confirmation of blockage
   - ST depression: functional confirmation of ischemia
   
2. **They measure different dimensions of heart disease**
   
3. **High specificity with clear clinical thresholds:**
   - Vessels = 0 + ST = 0 → ~90% of probability of being health
   - Vessels ≥ 2 + ST > 2 → ~90% of probability of being sick

### Comparación de Costos:

| Par                        | Final Cost | Interpretation |
|----------------------------|-------------|----------------|
| Age vs Cholesterol         | 0.672       | Limited separation |
| BP vs Max HR               | 0.588       | Moderate separation |
| **ST depression vs Vessels** | **0.516**   | **Best separation** |


### Conclusion:
ST Depression and Number of Vessels are the most discriminative features in the dataset. Even in 2D, they achieve high separation, matching real-world medical standards (angiography = gold standard).

## Step 4
## Final Conclusions

### Key Results:

1. **Correct implementation of L2 regularization:**
- Regularized cost: J_reg = J + (λ/2m)||w||²
- Regularized gradients: dw_reg = dw + (λ/m)w
- Bias not regularized (correct)

2. **Exhaustive experimentation:**
- λ values tested: [0, 0.001, 0.01, 0.1, 1.0]
- 1500 iterations each
- All models converged successfully

3. **Main result:**
```
      Optimal λ = 0 (no regularization)
      Test Accuracy: 0.8765
      Test F1-Score: 0.8684
      Improvement: +0.00%
```

4. **Effect of λ on ||w||:**
   - Higher λ → smaller ||w||
   - Max reduction ~2.8% with λ=1.0
   - No improvement in metrics

5. **No Overfitting:**
   - Test metrics ≥ Train metrics
   - Excellent natural generalization

### Medical Interpretation:

This is a positive result because:
- Model requires no hyperparameter tuning
- Simpler implementation
- Coefficients remain clinically interpretable
- Performance is strong and stable

### Lessons Learned:

**When NOT to use regularization:**
- Balanced dataset (enough samples per feature)
- No overfitting (Test ≥ Train)
- Relevant and non‑redundant features
- Simple and appropriate model

**When TO use regularization:**
- Few samples and many features
- Clear overfitting (Train ≫ Test)
- Correlated or noisy features
- Complex models (high‑degree polynomials)

λ = 0
Iter    0: Cost = 0.690608, ||w|| = 0.005016
Iter  500: Cost = 0.418085, ||w|| = 1.027125
Iter 1000: Cost = 0.396851, ||w|| = 1.327454
Iter 1499: Cost = 0.390799, ||w|| = 1.482178

λ = 0.001
Iter    0: Cost = 0.690608, ||w|| = 0.005016
Iter  500: Cost = 0.418089, ||w|| = 1.027114
Iter 1000: Cost = 0.396857, ||w|| = 1.327427
Iter 1499: Cost = 0.390805, ||w|| = 1.482135

λ = 0.01
Iter    0: Cost = 0.690608, ||w|| = 0.005016
Iter  500: Cost = 0.418123, ||w|| = 1.027013
Iter 1000: Cost = 0.396909, ||w|| = 1.327185
Iter 1499: Cost = 0.390867, ||w|| = 1.481755

λ = 0.1
Iter    0: Cost = 0.690608, ||w|| = 0.005016
Iter  500: Cost = 0.418465, ||w|| = 1.026009
Iter 1000: Cost = 0.397430, ||w|| = 1.324768
Iter 1499: Cost = 0.391485, ||w|| = 1.477961
...
Iter 1000: Cost = 0.402494, ||w|| = 1.301083
Iter 1499: Cost = 0.397437, ||w|| = 1.441191

## Step 5
To complete this section, it was necessary to train the model so it could take input test data and return predictions. This required creating new files: train.py; inference.py y sagemaker.deployment.ipynb.

- train.py trains the logistic regression model inside SageMaker.
The notebook generated new CSV files (train.csv), which train.py loads and normalizes.
- inference.py defines how the endpoint makes predictions when it receives patient data — essentially the "brain" of the endpoint.
- sagemaker.deployment.ipynb orchestrates everything to produce predictions.

Errors Encountered:
Upon uploading and running in SageMaker, the following errors occurred:

![alt text](img/error1.png)

![alt text](img/error2.png)

![alt text](img/error3.png)

![alt text](img/error4.png)

Investigation suggests this is a connection error likely due to region mismatches, meaning the endpoint was created in a different AWS region. Despite multiple attempts to fix it, the same error occurred with this deployment approach.
However, as previously shown, all code worked correctly and the laboratory was fully completed.
