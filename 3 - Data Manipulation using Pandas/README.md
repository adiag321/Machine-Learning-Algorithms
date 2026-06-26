# Pandas Data Analysis Cheat Sheet

## Table of Contents

1. [Reading Tabular Data](#1-reading-tabular-data)
2. [Selecting Data — iloc & loc](#2-selecting-data--iloc--loc)
3. [Basic Pandas Commands](#3-basic-pandas-commands)
4. [Rename Columns](#4-rename-columns)
5. [Remove Columns](#5-remove-columns)
6. [Sorting](#6-sorting)
7. [Filtering Data](#7-filtering-data)
8. [Iterating Series / DataFrame](#8-iterating-series--dataframe)
9. [Axis Parameter](#9-axis-parameter)
10. [String Methods](#10-string-methods)
11. [Change Datatypes](#11-change-datatypes)
12. [Groupby](#12-groupby)
13. [Exploring Categorical Columns](#13-exploring-categorical-columns)
14. [unique() and nunique()](#14-unique-and-nunique)
15. [CrossTab](#15-crosstab)
16. [Exploring a Numeric Series](#16-exploring-a-numeric-series)
17. [Handling Missing Values](#17-handling-missing-values)
18. [Indexing](#18-indexing)
19. [Creating Dummy Variables](#19-creating-dummy-variables)
20. [DateTime Operations](#20-datetime-operations)
21. [Remove Duplicate Rows](#21-remove-duplicate-rows)
22. [Apply Function](#22-apply-function)
23. [Aggregate Functions](#23-aggregate-functions)
24. [Rank, Shift, Cumsum](#24-rank-shift-cumsum)
25. [Rolling Window Functions](#25-rolling-window-functions)
26. [Concat](#26-concat)
27. [Merge DataFrames](#27-merge-dataframes)
28. [Joins](#28-joins)
29. [Append](#29-append)


---

## 1. Reading Tabular Data

```python
# Read a tab-separated file
df = pd.read_table('file.tsv')

# Read a CSV file
df = pd.read_csv('file.csv')

# Read CSV with custom separator, no header, and column names
df = pd.read_csv('file.csv', sep='|', header=None, names=['col1', 'col2'])

# Read CSV and set a column as the index
df = pd.read_csv('file.csv', index_col='column_name')
```

---

## 2. Selecting Data — iloc & loc

```python
# Select a single column (returns a Series)
df['column_name']
df.column_name                         # dot notation

# Select multiple columns
df[['col1', 'col2']]

# Filter columns by name pattern
df[[c for c in df.columns if 'keyword' in c]]

# Create a new column
df['new_col'] = df.col1 + ', ' + df.col2
```

### `loc` — label-based selection

```python
# Row 0, all columns
df.loc[0, :]                           

# Rows 0 through 2 (inclusive), all columns
df.loc[0:2, :]                         

# Rows 0-2, column 'City'
df.loc[0:2, 'City']                    

# Rows 0-2, columns 'City' and 'State'
df.loc[0:2, ['City', 'State']]         

# Rows 0-2, columns 'City' through 'State'
df.loc[0:2, 'City':'State']            

# Filtered rows, column 'State'
df.loc[df.City == 'Oakland', 'State']  
```

### `iloc` — integer-position-based selection

```python
# Rows at positions 0 & 1, columns at positions 0 & 3
df.iloc[[0, 1], [0, 3]]             

# Rows 0-1, columns 0-3 (end exclusive)
df.iloc[0:2, 0:4]                     

# Rows 0-1 (iloc preferred)
df[0:2]                             
```

---

## 3. Basic Pandas Commands

```python
# First 5 rows
df.head()

# First n rows
df.head(n)

# Last 5 rows
df.tail()

# Last n rows
df.tail(n)

# Summary statistics of numeric columns
df.describe()

# Summary statistics of string/object columns
df.describe(include=['object'])

# (rows, columns) tuple
df.dtypes

# DataFrame info + non-null counts + dtypes
df.info()

# List of column names
df.columns

# Index range
df.index

# Random sample of n rows
df.sample(n)

# Count of unique values in a Series
df.value_counts()

# Proportions instead of raw counts
df.value_counts(normalize=True)

# Select all numeric columns
df.select_dtypes(include=[np.number])
```

---

## 4. Rename Columns

```python
# Rename specific columns
df.rename(columns={'old_name': 'new_name', 'old2': 'new2'}, inplace=True)

# Rename all columns at once by assigning a list
df.columns = ['col1', 'col2', 'col3']

# Replace all spaces with underscores in column names
df.columns = df.columns.str.replace(' ', '_')

# Rename during file read using 'names' parameter
df = pd.read_csv('file.csv', header=0, names=['col1', 'col2'])
```

---

## 5. Remove Columns

```python
# Drop a single column
df.drop('column_name', axis=1, inplace=True)

# Drop multiple columns
df.drop(['col1', 'col2'], axis=1, inplace=True)

# Drop rows matching a condition
df.drop(df[df['col'] == 'value'].index, axis=0, inplace=True)

# Drop all non-numeric columns
df.drop(df.select_dtypes(include=[np.number]).columns, axis=1, inplace=True)
```

---

## 6. Sorting

```python
# Sort by a single column (ascending by default)
df.sort_values(by='column_name')

# Sort in descending order
df.sort_values(by='column_name', ascending=False)

# Sort a Series
df['column_name'].sort_values(ascending=False)

# Sort by multiple columns
df.sort_values(['col1', 'col2'])
```

---

## 7. Filtering Data

```python
# Filter rows by a condition
df[df.column_name >= 200]
df[df.column_name == 'value']

# Filter and select a single column
df[df.column_name >= 200]['other_column']
df.loc[df.column_name >= 200, 'other_column']  # preferred

# Filter with AND condition (&)
df[(df.col1 >= 200) & (df.col2 == 'Drama')]

# Filter with OR condition (|)
df[(df.col1 == 'Crime') | (df.col1 == 'Drama') | (df.col1 == 'Action')]

# Filter using isin() — check membership in a list
df[df.col1.isin(['Crime', 'Drama', 'Action'])]

# Combine isin() with another condition
df[(df['col1'].isin(['Crime'])) & (df['col2'] > 200)]
```

---

## 8. Iterating Series / DataFrame

```python
# Iterate through a Series (like a list)
for value in df['column_name']:
    print(value)

# Iterate through DataFrame rows
for index, row in df.iterrows():
    print(index, row.col1, row.col2)
```

---

## 9. Axis Parameter

```python
# axis=0 (default) → operates across rows (down each column)
df.mean()
df.mean(axis=0)
df.mean(axis='index')   # alias for axis=0

# axis=1 → operates across columns (across each row)
df.mean(axis=1)
df.mean(axis='columns') # alias for axis=1

# Drop using axis
df.drop(['col1', 'col2'], axis='columns', inplace=True)  # drop columns
df.drop(index_list, axis='index', inplace=True)           # drop rows
```

---

## 10. String Methods

```python
# All string methods are accessed via .str accessor
df['col'].str.upper()                          # Convert to uppercase
df['col'].str.lower()                          # Convert to lowercase
df['col'].str.len()                            # Length of each string
df['col'].str.strip()                          # Strip whitespace
df['col'].str.replace('old', 'new')            # Replace substring
df['col'].str.replace('[\\[\\]]', '')          # Replace with regex
df['col'].str.contains('keyword')              # Boolean: contains substring
df['col'].str.startswith('prefix')             # Boolean: starts with
df['col'].str.endswith('suffix')               # Boolean: ends with
df['col'].str.split('delimiter')               # Split into list
df['col'].str.replace(' ', '_')                # Replace space with underscore

# Use boolean result to filter rows
df[df['col'].str.contains('keyword')]

# Chain string methods
df['col'].str.replace('[', '').str.replace(']', '')
```

---

## 11. Change Datatypes

```python
# View current data types
df.dtypes

# Convert a column to a different type
df['col'] = df['col'].astype(float)
df['col'] = df['col'].astype(int)
df['col'] = df['col'].astype(str)

# Convert string column to numeric (e.g. remove '$' first)
df['price'].str.replace('$', '').astype(float)

# Convert boolean Series to integer (False=0, True=1)
df['col'].str.contains('keyword').astype(int)

# Convert to datetime
df['date_col'] = pd.to_datetime(df['date_col'])
```

---

## 12. Groupby

```python
# Group by one column and apply an aggregation
df.groupby('col')['numeric_col'].mean()
df.groupby('col')['numeric_col'].sum()
df.groupby('col')['numeric_col'].max()
df.groupby('col')['numeric_col'].min()
df.groupby('col')['numeric_col'].count()

# Group by one column, apply to all numeric columns
df.groupby('col').mean()

# Group by multiple columns
df.groupby(['col1', 'col2'])['numeric_col'].mean()

# agg() — apply multiple aggregation functions at once
df.groupby('col')['numeric_col'].agg(['min', 'max', 'mean', 'count'])

# Reset index after groupby
df.groupby('col')['numeric_col'].mean().reset_index()
```

---

## 13. Exploring Categorical Columns

```python
# Get value counts of a categorical column
df['col'].value_counts()

# Get proportions
df['col'].value_counts(normalize=True)

# Describe a non-numeric (object) column
df['col'].describe()
```

---

## 14. unique() and nunique()

```python
# Get all unique values in a Series
df['col'].unique()

# Count the number of unique values
df['col'].nunique()

# Count unique values per column across the whole DataFrame
df.nunique()
```

---

## 15. CrossTab

```python
# Cross-tabulation of two columns (frequency table)
pd.crosstab(df['col1'], df['col2'])

# With normalization (show proportions)
pd.crosstab(df['col1'], df['col2'], normalize=True)

# With margins (row/column totals)
pd.crosstab(df['col1'], df['col2'], margins=True)
```

---

## 16. Exploring a Numeric Series

```python
df['col'].mean()        # Mean
df['col'].median()      # Median
df['col'].std()         # Standard deviation
df['col'].var()         # Variance
df['col'].min()         # Minimum value
df['col'].max()         # Maximum value
df['col'].sum()         # Sum
df['col'].count()       # Count of non-null values
df['col'].describe()    # All summary stats at once
df['col'].quantile(0.25)  # 25th percentile
df['col'].hist()        # Histogram plot
df['col'].plot(kind='hist', bins=20)
```

---

## 17. Handling Missing Values

```python
# Detect missing values
df.isnull()                            # Boolean DataFrame — True where NaN
df.isnull().sum()                      # Count of nulls per column
df.isnull().sum().sum()                # Total null count across entire DataFrame
df.notnull()                           # Inverse of isnull()

# Drop rows/columns with missing values
df.dropna()                            # Drop rows containing any NaN
df.dropna(axis=1)                      # Drop columns containing any NaN
df.dropna(subset=['col1', 'col2'])     # Drop rows where specific columns are NaN
df.dropna(how='all')                   # Drop rows where ALL values are NaN

# Fill missing values
df.fillna(value)                       # Fill all NaNs with a value
df['col'].fillna(df['col'].mean())     # Fill NaNs with column mean
df.fillna(method='ffill')              # Forward-fill (propagate last valid value)
df.fillna(method='bfill')              # Back-fill (propagate next valid value)
```

---

## 18. Indexing

```python
# Set a column as the index
df = df.set_index('column_name')
df.set_index('column_name', inplace=True)

# Reset the index (move index back to a column)
df.reset_index(inplace=True)

# Access the index
df.index

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# Read CSV with index column
df = pd.read_csv('file.csv', index_col='column_name')
```

---

## 19. Creating Dummy Variables

```python
# Create dummy (one-hot encoded) variables for a column
pd.get_dummies(df['col'])

# Add prefix to dummy column names
pd.get_dummies(df['col'], prefix='prefix')

# Create dummies for all object columns in a DataFrame
pd.get_dummies(df)

# Drop first dummy column to avoid multicollinearity
pd.get_dummies(df['col'], drop_first=True)
```

---

## 20. DateTime Operations

```python
# Convert a column to datetime
df['date_col'] = pd.to_datetime(df['date_col'])

# Extract date components
df['date_col'].dt.year
df['date_col'].dt.month
df['date_col'].dt.day
df['date_col'].dt.hour
df['date_col'].dt.minute
df['date_col'].dt.dayofweek    # Monday=0, Sunday=6
df['date_col'].dt.day_name()   # e.g. 'Monday'
df['date_col'].dt.date         # Date portion only

# Calculate date differences
(df['end_date'] - df['start_date']).dt.days

# Filter by date range
df[df['date_col'] >= '2023-01-01']
df[(df['date_col'] >= '2023-01-01') & (df['date_col'] <= '2023-12-31')]
```

---

## 21. Remove Duplicate Rows

```python
# Identify duplicate rows
df.duplicated()                        # Boolean Series — True for duplicates
df.duplicated().sum()                  # Count of duplicate rows

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop duplicates based on specific columns
df.drop_duplicates(subset=['col1', 'col2'], inplace=True)

# Keep last occurrence instead of first
df.drop_duplicates(keep='last', inplace=True)
```

---

## 22. Apply Function

```python
# Apply a function to each element in a Series
df['col'].apply(function_name)
df['col'].apply(lambda x: x * 2)

# Apply a function to each column (axis=0) or row (axis=1)
df.apply(function_name)
df.apply(function_name, axis=1)

# Apply a function to each element in the entire DataFrame
df.applymap(function_name)

# Map values using a dictionary
df['col'].map({'A': 1, 'B': 2, 'C': 3})
```

---

## 23. Aggregate Functions

```python
# Common aggregations on a column
df['col'].sum()
df['col'].mean()
df['col'].median()
df['col'].min()
df['col'].max()
df['col'].count()
df['col'].std()

# Multiple aggregations using agg()
df.groupby('col').agg({'num_col1': 'sum', 'num_col2': 'mean'})
df.groupby('col')['num_col'].agg(['min', 'max', 'mean', 'count'])

# Aggregate the entire DataFrame
df.sum()
df.mean()
df.min()
df.max()

# Pivot Tables
# Create a pivot table
pivot = df.pivot_table(index='col1', columns='col2', values='num_col', aggfunc='sum')

# Or with multiple aggregations
pivot = df.pivot_table(index='col1', columns='col2', values='sales', aggfunc=['sum', 'mean', 'count'])
```

---

## 24. Rank, Shift, Cumsum

```python
# Rank values in a column
df['col'].rank()
df['col'].rank(ascending=False)        # Rank in descending order
df['col'].rank(method='dense')         # Dense ranking (no gaps)

# Shift values by n periods
df['col'].shift(1)                     # Shift down by 1 (lag)
df['col'].shift(-1)                    # Shift up by 1 (lead)

# Cumulative sum
df['col'].cumsum()

# Cumulative maximum
df['col'].cummax()

# Cumulative minimum
df['col'].cummin()

# Cumulative product
df['col'].cumprod()
```

---

## 25. Rolling Window Functions

```python
# Rolling mean (moving average) over n periods
df['col'].rolling(window=n).mean()

# Rolling sum
df['col'].rolling(window=n).sum()

# Rolling standard deviation
df['col'].rolling(window=n).std()

# Rolling min / max
df['col'].rolling(window=n).min()
df['col'].rolling(window=n).max()

# Expanding window (cumulative from start)
df['col'].expanding().mean()
```

---

## 26. Concat

```python
# Concatenate DataFrames vertically (stack rows)
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)   # Reset index

# Concatenate DataFrames horizontally (add columns)
pd.concat([df1, df2], axis=1)

# Concatenate with keys to create MultiIndex
pd.concat([df1, df2], keys=['df1', 'df2'])
```

---

## 27. Merge DataFrames

```python
# Inner merge (default) — only matching rows
pd.merge(df1, df2, on='key_column')

# Merge on different column names
pd.merge(df1, df2, left_on='col_df1', right_on='col_df2')

# Merge on index
pd.merge(df1, df2, left_index=True, right_index=True)

# Merge on multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])
```

---

## 28. Joins

```python
# Inner join — rows present in both DataFrames
pd.merge(df1, df2, on='key', how='inner')

# Left join — all rows from df1, matching from df2
pd.merge(df1, df2, on='key', how='left')

# Right join — all rows from df2, matching from df1
pd.merge(df1, df2, on='key', how='right')

# Outer join — all rows from both DataFrames
pd.merge(df1, df2, on='key', how='outer')

# Join using DataFrame .join() method (joins on index by default)
df1.join(df2)
df1.join(df2, how='left')
df1.join(df2, on='key_col')
```

## 29. Append

```python
# Append rows from one DataFrame to another (creates new DataFrame)
df_new = pd.concat([df1, df2], ignore_index=True)
df_new = pd.concat([df1, df2])  # Keep original index

# Or using the append() method (older syntax, returns new DataFrame)
df_new = df1.append(df2)

# Append with ignore_index to reset index
df_new = df1.append(df2, ignore_index=True)
```

---

## Quick Reference

| Task | Syntax |
|------|--------|
| Read CSV | `pd.read_csv('file.csv')` |
| First N rows | `df.head(n)` |
| Last N rows | `df.tail(n)` |
| Shape | `df.shape` |
| Column types | `df.dtypes` |
| Summary stats | `df.describe()` |
| Null counts | `df.isnull().sum()` |
| Drop nulls | `df.dropna()` |
| Fill nulls | `df.fillna(value)` |
| Sort | `df.sort_values('col')` |
| Filter | `df[df['col'] > value]` |
| Group & aggregate | `df.groupby('col')['num'].mean()` |
| Rename columns | `df.rename(columns={'old': 'new'})` |
| Drop column | `df.drop('col', axis=1)` |
| New column | `df['new'] = expression` |
| Unique values | `df['col'].unique()` |
| Value counts | `df['col'].value_counts()` |
| Apply function | `df['col'].apply(func)` |
| Merge | `pd.merge(df1, df2, on='key')` |
| Concat | `pd.concat([df1, df2])` |
| Dummy variables | `pd.get_dummies(df['col'])` |
| Convert type | `df['col'].astype(float)` |
| To datetime | `pd.to_datetime(df['col'])` |
| Rolling mean | `df['col'].rolling(3).mean()` |
| Cumulative sum | `df['col'].cumsum()` |


