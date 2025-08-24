This code analyzes the complete set of EO-to-DR solutions to determine an optimal search strategy usable in an FMC attempt.

See https://docs.google.com/document/d/1AUQVrecEFZyR6okXAyJgtQZA-A0AJmAUevhJZlQMjAs/edit?usp=sharing

# Generating the raw data
```
g++ -Wall gen_full_data.cpp
./a.out

python split_into_cases.py
```

# Generating the reports

## Findability breakdowns
```commandline
python stats.py mutual-info
python stats.py findability-all
python stats.py findability-top10
python stats.py findability-4c4e
```

## Decision tree training
```commandline
python tree.py <n_bad_corners> <n_bad_edges> <objective-move-count> [easy|findable|hard]
```
