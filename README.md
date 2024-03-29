# SVM-Trainer
SVM trainer based on Zoutendijk optimization algorithm, only for learning, not for commercial use.

## How to run?

Open the bash and change to the project directory, run

```bash
python -u svm.py
```

Change the variables defined in file `svm_params.py` to change the distribution of two classes of samples.

## What's the point? 

SVM (Support Vector Machine) is a classic machine-learning algorithm, usually we can implement a SVM model using Python modules conveniently. These modules hide a lot of details about how to train a SVM model by mathematical methods. 

The basic SVM problem can be transformed to a **Convex optimization problem**, there are a lot of optimization methods to solve it. This project adopts Zoutendijk method to solve basic soft margin SVM problems and was proved performing well even when two classes of samples are apparently overlapped.

