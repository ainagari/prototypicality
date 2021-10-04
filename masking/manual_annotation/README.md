

## Manual annotation

This directory contains the manual annotations of the `bert-large-uncased` predictions (most+plural template) for 90 nouns in MRD. This is described in more detail in Sections 4.1 and Appendix B.4.

`annotations_A1.csv` and `annotations_A2.csv` contain the annotations of A#1 and A#2. "1" indicates that the prediction describes some property of the noun, and "0" means that it does not.

The **agreement** can be calculated with:

`python agreement.py annotations_A1.csv annotations_A2.csv`

The **number of correct properties @k** can be calculated with:

`python evaluation.py annotations_A1.csv annotations_A2.csv`

