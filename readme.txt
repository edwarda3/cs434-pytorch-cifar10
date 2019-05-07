CS 434, Implementation Assignment 3
Alex Edwards, Jacob Dugan

Ensure that Python 3.5+ is used,
PyTorch, torchvision, numpy, scipy, and matplotlib are installed,
and that the cifar-10-batches-py/ folder is in the same folder as the python files.

---

For ease of testing, the runner.py file will run all questions with various sets of parameters.

Note that this takes a while, as it goes through ~50 models.

$ python3 runner.py

To change the epochs, change the epochs variable in each of the question files above the train() function.

---

To run each question,

$ python3 q1.py <learning_rate>

$ python3 q2.py <learning_rate>

$ python3 q3.py <dropout> <momentum> <weight_decay>

$ python3 q4.py

Note that all parameters passed in are floating point numbers.