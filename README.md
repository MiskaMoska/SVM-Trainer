# SVM-Trainer
SVM trainer based on Zoutendijk optimization algorithm, only for learning, not for commercial use.

## How to run?

Open your bash and change to the project directory, run

```bash
python -u svm.py
```

Change file `svm_params.py` to change the distribution of two classes of samples

## What's the point? 

SVM (Support Vector Machine) is a popular machine-learning algorithm, usually we can implement a SVM model using Python modules conveniently. These modules hide a lot of details about how to train a SVM model by mathematical methods. 

The basic SVM problem can be transformed to a **Convex optimization problem** by some ways, there are lots of ways or methods to solve it. This project adopts Zoutendijk method to solve basic Soft Margin SVM problems and was proved performing well even when two classes of samples show apparent overlap.

