# Everything about YAML in Machine Learning

<b>YAML: Yet Another Markup Language</b>
YAML file formats have become a crowd favorite for configurations, presumably for their ease of readability. YAML is relatively easy to write. Within simple YAML files, there are no data formatting items, such as braces and square brackets; most of the relations between items are defined using indentation.

### Basics Syntax of YAML file
##### (a) Comments
In YAML file comments begin with a pound sign.
Example:
```
# my first comment
```
##### (b) key-value Pair
Datatype in YAML is in the form of key-value pairs like other programming languages such as Python, Perl, and javascript. The key is always a string and the value can be any datatype.

Example:
```
learning_rate: 0.1
evaluation_metric: rmse
```

##### (c) Numerical Data
YAML recognizes and support different numerical data type such as integer, decimal, hexadecimal, or octal.
Example:
```
test_size: 0.2
epochs: 50
scientific_notation: 1e+12
```

##### (d) String
Write string in YAML is very simple and you don’t have to specify them in quotes. However, they can be.

Example:
```
experiment_title: find the best model by using f1 score
```

##### (e) Boolean
YAML indicates boolean values with the keywords True, On and Yes for true, and false is indicated with False, Off, or No.
Example:
```
cross_validation: True
save_model: False
```

##### (f) Array
YAML supports the creation of arrays or lists on a single line.
Example:
```
ages: [24,76,45,21,45]
labels: ["class_one","class_two”,"class_three"]
```

### Rules for Creating YAML file
When it’s come to creating a YAML file, you have to follow some very important basic rules.

* The files should have .yaml as the extension.
* YAML is case sensitive.
* Do not use tabs while creating YAML files.