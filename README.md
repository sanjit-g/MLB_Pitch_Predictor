# Pitch Predictor

A machine-learning experiment that uses MLB Statcast data to recommend a **pitch type** and **target zone** based on the current game situation.

The notebook explores several modeling approaches before arriving at a multi-output neural network. Rather than simply predicting what pitch was historically thrown, the later versions first estimate which pitch/zone combinations produced the best outcomes for a given game state and then train a model to reproduce those recommendations.

## Project Goal

Given contextual information such as:

* Ball-strike count
* Number of outs
* Base-runner configuration
* Batter handedness
* Pitcher handedness
* Pitch number in the plate appearance

the model returns:

1. A recommended **pitch type**
2. A recommended **Statcast zone**
3. Confidence/probability estimates for each prediction

The notebook also supports **pitcher arsenal masking**, which prevents the final recommendation from selecting a pitch that a particular pitcher does not throw.



## Notebook Overview

The notebook contains multiple iterations of the project:

* **Version 1:** Multi-output Random Forest that directly predicts pitch type and zone.
* **Version 2:** Begins deriving an "optimal" pitch/zone from historical outcome data.
* **Version 3:** Expands the scoring logic, but identifies a data-leakage problem.
* **Version 4:** Splits the data before generating optimal labels.
* **Version 5:** Uses a `HistGradientBoostingClassifier`.
* **Version 6:** Replaces the tree model with a multi-output neural network and evaluates the recommendations on 2024 data.

The final section of the notebook is the most current implementation.



## Data

The project uses pitch-level data from **MLB Statcast**, accessed through the `pybaseball` package.

The final workflow expects:

```text
statcast_2023.csv
statcast_2024.csv
```

The notebook includes an example Statcast download:

```python
from pybaseball import statcast

data = statcast(
    start_dt="2024-03-30",
    end_dt="2024-10-01"
)

data.to_csv("statcast_2024.csv", index=False)
```

The same approach can be used with a different date range to generate the 2023 dataset.

### Statcast Columns Used

The final model uses or derives information from:

```text
pitcher
pitch_type
zone
balls
strikes
inning
outs_when_up
on_1b
on_2b
on_3b
stand
p_throws
pitch_number
estimated_woba_using_speedangle
description
events
```

Rows missing pitch type, zone, estimated wOBA, pitch description, or event information are removed.



## Feature Engineering

Base-runner columns are converted to binary indicators:

```text
on_1b
on_2b
on_3b
```

Batter and pitcher handedness are one-hot encoded into features such as:

```text
stand_R
p_throws_R
```

The final neural network is trained with the following inputs:

```text
balls
strikes
pitch_number
outs_when_up
on_1b
on_2b
on_3b
stand_R
p_throws_R
```



## Defining an "Optimal" Pitch

Instead of labeling a situation using only the pitch that was actually thrown, the later versions create an outcome score for each historical **pitch type + zone** combination within a game context.

Contexts are grouped by:

```text
balls
strikes
outs_when_up
on_1b
on_2b
on_3b
stand_R
p_throws_R
```

Several outcome variables are calculated.

### Strike Score

The following pitch descriptions count as positive strike outcomes:

```text
called_strike
swinging_strike
foul_tip
swinging_strike_blocked
foul
```

### Ball Penalty

A pitch described as a ball receives a penalty of `0.1`.

### Out Reward

The following events receive an out reward:

```text
field_out
force_out
double_play
strikeout
```

### Hit Penalty

The following events receive a hit penalty of `0.2`:

```text
single
double
triple
home_run
```

### Combined Score

The notebook calculates:

```text
combined_score =
    0.5 * (1 - strike_score)
  + 0.3 * estimated_woba_using_speedangle
  + 0.1 * hit_penalty
  + 0.1 * ball_penalty
  - 0.1 * out_reward
```

A **lower score is considered better**.

For each game context, the pitch type and zone with the lowest historical combined score become the training labels.

Importantly, the later notebook versions split the dataset **before** deriving these optimal labels so that the test set is not directly used to determine the recommended pitch/zone combination.



## Neural Network

The final model is a shared multi-output neural network implemented with TensorFlow/Keras.

### Architecture

```text
Input features
     |
Dense(64, ReLU)
     |
Dense(64, ReLU)
     |
     +-------------------+
     |                   |
Pitch Type Softmax   Zone Softmax
```

Both outputs are classification tasks.

The model is compiled with:

* **Optimizer:** Adam
* **Loss:** Sparse categorical cross-entropy
* **Training epochs:** 10
* **Batch size:** 64

The notebook also performs 5-fold validation and plots confusion matrices for pitch and zone predictions.



## Pitcher Arsenal Masking

The neural network initially assigns probabilities across every pitch type in its label set.

The notebook can then restrict the recommendation to pitches in a specific pitcher's arsenal:

```python
pitcher_arsenal = ['FF', 'SL', 'CH', 'CU', 'SI']
```

Probabilities for pitches outside the arsenal are set to zero before the final pitch is selected.

This allows the same model to produce recommendations that are more realistic for an individual pitcher.


## Zone Visualization

Predicted Statcast-zone probabilities are mapped onto an 8×8 grid and displayed as a heatmap.

This provides a visual representation of where the model recommends locating the next pitch instead of returning only the numerical Statcast zone.



## 2024 Evaluation

The model is trained using the 2023 workflow and is then applied to `statcast_2024.csv`.

For each predicted pitch/zone combination, the notebook examines average:

* Strike score
* Estimated wOBA
* Hit penalty
* Out reward

The notebook also evaluates whether its recommendation matches pitches associated with favorable 2024 outcomes.

In the current saved notebook run:

```text
Strict Match Accuracy (pitch AND zone): 0.54%
Relaxed Match Accuracy (pitch OR zone): 16.86%
```

These values are much lower than the neural network's training accuracy, which reached approximately:

```text
Pitch output training accuracy: 96.0%
Zone output training accuracy: 95.3%
```

The difference is important: high accuracy on the generated training labels does **not** necessarily mean that the model has learned a strong real-world pitch-selection strategy.



## Installation

Clone the repository and install the required packages.

```bash
pip install pybaseball pandas numpy scikit-learn matplotlib tensorflow
```

The notebook can then be run in:

* Jupyter Notebook
* JupyterLab
* Google Colab
* VS Code with the Jupyter extension



## Running the Notebook

The notebook can then be run in:

* Jupyter Notebook
* JupyterLab
* Google Colab
* VS Code with the Jupyter extension

1. Install the required Python packages.(see below)
2. Place `statcast_2023.csv` in the notebook's working directory.
3. Place or generate `statcast_2024.csv` for out-of-season evaluation.
4. Open the notebook.
5. Run the cells in order.
6. For the current implementation, use the **Version 6 - Neural Net** section.
7. Modify the `scenario` dictionary to represent the game situation you want to evaluate.
8. Modify `pitcher_arsenal` to contain only the pitches available to the pitcher.

 If you are running on your local machine, make sure you have the required packages installed:
```bash
pip install pybaseball pandas numpy scikit-learn matplotlib tensorflow
```

Example scenario:

```python
scenario = pd.DataFrame([{
    'balls': 3,
    'strikes': 2,
    'outs_when_up': 1,
    'on_1b': 0,
    'on_2b': 0,
    'on_3b': 0,
    'stand_R': 1,
    'p_throws_R': 0,
    'pitch_number': 5
}])
```

Example arsenal:

```python
pitcher_arsenal = ['FF', 'SL', 'CH', 'CU', 'SI']
```

The model then prints the pitch probabilities and produces a recommendation similar to:

```text
Suggested Pitch: SL
Suggested Zone: 3
Suggested Pitch (with arsenal masking): SL
```

---

## Pitch-Type Abbreviations

| Code | Pitch              |
| ---- | ------------------ |
| `FF` | Four-Seam Fastball |
| `SI` | Sinker             |
| `FC` | Cutter             |
| `SL` | Slider             |
| `ST` | Sweeper            |
| `CU` | Curveball          |
| `KC` | Knuckle Curve      |
| `CH` | Changeup           |
| `FS` | Split-Finger       |
| `SV` | Slurve             |

Additional Statcast pitch classes may also appear in the dataset.

---

## Current Limitations

This notebook is an experimental prototype and should not yet be interpreted as a validated pitch-calling system.

### Generated Labels

The model does not learn directly from a coach-defined ground truth. Its "optimal" pitch labels are generated from the notebook's custom weighted scoring function. Changing those weights can change the recommended strategy substantially.

### Limited Game Context

The final feature set does not currently include many factors that could affect pitch selection, including:

* Pitcher identity
* Batter identity
* Pitcher-specific arsenal quality
* Previous pitch type and location
* Pitch velocity or movement
* Batter strengths and weaknesses
* Platoon-specific historical matchups
* Score differential
* Leverage
* Times through the order
* Fatigue
* Pitch sequencing

### Sparse Context Groups

Some game states and pitch/zone combinations have many more observations than others. A historically strong result from a small sample may therefore be selected as the optimal label.

### Training Accuracy vs. Real-World Performance

The neural network learns labels produced from context groups in the training data. As a result, classification accuracy against those labels should not be treated as direct evidence that the recommendation would improve MLB outcomes.

The much lower 2024 strict/relaxed match results indicate that out-of-season generalization remains a major area for improvement.

### Validation Implementation

The current K-fold section repeatedly trains the same neural-network instance across folds rather than constructing a fresh model for every fold. For fully independent cross-validation, a new model should be initialized for each fold.

### Label Encoding

Pitch and zone encoders are currently fitted while processing both the training and test DataFrames. A production version should fit encoders on the training labels once and use those same mappings for validation/test data.

---

## Possible Next Steps

Potential improvements include:

* Add pitcher and batter identities or embeddings
* Model each pitcher's actual arsenal automatically
* Include the previous pitch and pitch sequence
* Add velocity, spin, horizontal movement, and vertical movement
* Include batter-specific performance by pitch type and location
* Use minimum sample-size thresholds when generating optimal labels
* Apply shrinkage or Bayesian estimates to sparse pitch/context groups
* Tune the weights used in the combined outcome score
* Compare recommendations against simpler baselines
* Use time-based train/validation/test splits
* Rebuild the K-fold procedure with a fresh model per fold
* Calibrate predicted probabilities
* Evaluate expected run value or change in run expectancy
* Train on multiple seasons and reserve a later season strictly for testing


