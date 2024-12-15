import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('cleaned_la_crime_data.csv')

# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe(include='all'))

# Correlation Matrix
print("\nCorrelation Matrix:")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# Visualizations
sns.set(style="whitegrid")

# Distribution of Crime Counts by Area
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Large Area', order=df['Large Area'].value_counts().index)
plt.title('Distribution of Crime Counts by Area')
plt.xlabel('Large Area')
plt.ylabel('Crime Count')
plt.xticks(rotation=45)
plt.savefig('image/crime_counts_by_area.png')

# Distribution of Crime Counts by Small Area
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='AREA NAME', order=df['AREA NAME'].value_counts().index)
plt.title('Distribution of Crime Counts by Area')
plt.xlabel('Small Area')
plt.ylabel('Crime Count')
plt.xticks(rotation=45)
plt.savefig('image/crime_counts_by_area_small.png')


# Distribution of Crime Counts by Day of Week
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Day of Week', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Distribution of Crime Counts by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Crime Count')
plt.savefig('image/crime_counts_by_day_of_week.png')

# Distribution of Crime Counts by Hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Hour')
plt.title('Distribution of Crime Counts by Hour')
plt.xlabel('Hour')
plt.ylabel('Crime Count')
plt.savefig('image/crime_counts_by_hour.png')

# Heatmap of Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.savefig('image/correlation_matrix_heatmap.png')

# For each day of the week, plot the distribution of crime counts by hour
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Hour', hue='Day of Week')
plt.title('Distribution of Crime Counts by Hour for Each Day of the Week')
plt.xlabel('Hour')
plt.ylabel('Crime Count')
plt.legend(title='Day of Week')
plt.savefig('image/crime_counts_by_hour_for_each_day.png')

# For each day of the week, plot the distribution of crime counts by hour
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(12, 42))  # Adjust the height to fit all subplots

for i, day in enumerate(days_of_week):
    plt.subplot(7, 1, i + 1)
    sns.countplot(data=df[df['Day of Week'] == day], x='Hour')
    plt.title(f'Distribution of Crime Counts by Hour on {day}')
    plt.xlabel('Hour')
    plt.ylabel('Crime Count')

plt.tight_layout()
plt.savefig('image/crime_counts_by_hour_for_each_day_separated.png')