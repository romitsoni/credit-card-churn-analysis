1. Original dataset
--------------------------------------------------
Dataset shape: (1010, 10)
Columns: ['CustomerID', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Churn']

Database info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1010 entries, 0 to 1009
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerID       1010 non-null   object 
 1   Gender           1004 non-null   object 
 2   Age              1008 non-null   float64
 3   Tenure           1007 non-null   float64
 4   Balance          1006 non-null   float64
 5   NumOfProducts    1006 non-null   float64
 6   HasCrCard        1008 non-null   object 
 7   IsActiveMember   1005 non-null   object 
 8   EstimatedSalary  1009 non-null   float64
 9   Churn            1007 non-null   object 
dtypes: float64(5), object(5)
memory usage: 79.0+ KB


2. First 10 records of the dataset
--------------------------------------------------
CustomerID Gender  Age  Tenure   Balance  NumOfProducts HasCrCard IsActiveMember  EstimatedSalary Churn
  CUST0001   Male 56.0     4.0      0.00            4.0       0.0            0.0         40282.42   1.0
  CUST0002    NaN 28.0     8.0  67408.01            4.0       0.0            1.0         27333.51   0.0
  CUST0003 Female 47.0     6.0   1154.97            1.0       0.0            1.0         99514.91   1.0
  CUST0004   Male 42.0     1.0      0.00            2.0       1.0            1.0        146588.22   0.0
  CUST0005   Male 64.0     3.0  77109.94            4.0       0.0            0.0        131792.25   0.0
  CUST0006   Male 26.0     7.0      0.00            4.0       1.0            1.0         70104.15   1.0
  CUST0007   Male 19.0     4.0  48964.07            3.0       1.0            0.0        128315.34   1.0
  CUST0008   Male 34.0     4.0  37264.98            2.0       1.0            1.0         47032.42   0.0
  CUST0009 Female 46.0     2.0 155251.43            2.0       1.0            1.0         97727.00   0.0
  CUST0010   Male 25.0     7.0 104646.02            4.0       0.0            1.0        117151.61   0.0

3. Missing values analysis
--------------------------------------------------
Missing values per column:
Gender: 6 (0.59%)
Age: 2 (0.20%)
Tenure: 3 (0.30%)
Balance: 4 (0.40%)
NumOfProducts: 4 (0.40%)
HasCrCard: 2 (0.20%)
IsActiveMember: 5 (0.50%)
EstimatedSalary: 1 (0.10%)
Churn: 3 (0.30%)

Total missing values: 30
