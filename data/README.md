# Data

## Breast Cancer Wisconsin (Diagnostic)

- `Cancer_Data.csv`: the dataset used in the thesis (569 rows, 33 columns).
- `wdbc.names`: the original description of the dataset by its creators.

**Source:** W. H. Wolberg, W. N. Street, O. L. Mangasarian. *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository, 1995. https://doi.org/10.24432/C5DW2B

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

## Columns

| Column | Description |
|---|---|
| `id` | sample identifier (not a feature) |
| `diagnosis` | target: `M` = malignant, `B` = benign |
| `<feature>_mean` | mean value of the feature over the nuclei in the image |
| `<feature>_se` | standard error of the feature |
| `<feature>_worst` | mean of the three largest values of the feature |
| `Unnamed: 32` | empty column, removed during cleaning |

The ten features computed for each cell nucleus are: `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, `fractal_dimension`.

There are no missing values. Class distribution: 357 benign (62.7%), 212 malignant (37.3%).
