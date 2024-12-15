import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# Step 1: Load the Dataset
dataset_path = '../../Data/crime_dataset.csv'
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    print("Failed to load data. Error:", e)
    exit()

# Step 2: Basic Information about the Data
def basic_info(df):
    # Missing Values per Column
    plt.figure(figsize=(20, 15))
    missing_values = df.isnull().sum()
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.xlabel('Column Names')
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Values per Column')
    plt.xticks(rotation=90)
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    # Basic Statistics (excluding record identifier columns and first numerical feature)
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    if 'record_id' in numerical_features:
        numerical_features.remove('record_id')
    if len(numerical_features) > 0:
        numerical_features.pop(0)  # Remove the first numerical feature
    df[numerical_features].describe().plot(kind='box', figsize=(20, 15))
    plt.title('Boxplot of Numerical Features')
    plt.xticks(range(len(numerical_features)), numerical_features, rotation=45)
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    output_dir = dataset_path.split('.')[0] + '_foto'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Description of Dataset Columns
    for col in df.columns:
        max_value = df[col].max() if df[col].dtype in ['int64', 'float64'] else 'N/A'
        min_value = df[col].min() if df[col].dtype in ['int64', 'float64'] else 'N/A'
        mean_value = df[col].mean() if df[col].dtype in ['int64', 'float64'] else 'N/A'
        std_dev = df[col].std() if df[col].dtype in ['int64', 'float64'] else 'N/A'
        sample_values = df[col].dropna().sample(5, random_state=1).tolist() if df[col].notna().sum() >= 5 else df[col].dropna().tolist()
        description = (f"Column: {col}\n"
                       f"Data Type: {df[col].dtype}\n"
                       f"Number of Non-Null Values: {df[col].notnull().sum()}\n"
                       f"Number of Unique Values: {df[col].nunique()}\n"
                       f"Max Value: {max_value}\n"
                       f"Min Value: {min_value}\n"
                       f"Mean Value: {mean_value}\n"
                       f"Standard Deviation: {std_dev}\n"
                       f"Sample Values: {sample_values}")
        img = Image.new('RGB', (800, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), description, fill=(0, 0, 0), font=font)
        output_file = os.path.join(output_dir, f"{col}.png")
        img.save(output_file)

# Step 3: Exploratory Data Analysis (EDA)
def exploratory_analysis(df):
    # Update column names to be consistent for analysis
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Distribution of crimes by year
    df['date_occ'] = pd.to_datetime(df['date_occ'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')  # Convert date column to datetime format
    df.dropna(subset=['date_occ'], inplace=True)  # Remove rows with invalid date formats
    df['year'] = df['date_occ'].dt.year
    crime_by_year = df['year'].value_counts().sort_index()
    plt.figure(figsize=(20, 15))
    sns.barplot(x=crime_by_year.index, y=crime_by_year.values)
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.title('Number of Crimes by Year')
    plt.xticks(crime_by_year.index, rotation=45)
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    # Most common crime types
    plt.figure(figsize=(20, 15))
    crime_types = df['crm_cd_desc'].value_counts().head(10)
    sns.barplot(x=crime_types.values, y=crime_types.index)
    plt.xlabel('Number of Incidents')
    plt.ylabel('Crime Type')
    plt.title('Top 10 Most Common Crime Types')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.3)
    plt.show()

    # Crimes by area
    plt.figure(figsize=(20, 15))
    sns.countplot(y='area_name', data=df, order=df['area_name'].value_counts().index)
    plt.xlabel('Number of Crimes')
    plt.ylabel('Area Name')
    plt.title('Crime Count by Area')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.3)
    plt.show()

    # Victim Age Distribution
    plt.figure(figsize=(20, 15))
    sns.histplot(df['vict_age'], bins=30, kde=True)
    plt.xlabel('Victim Age')
    plt.title('Victim Age Distribution')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

# Step 4: Temporal Analysis of Crime Patterns
def temporal_analysis(df):
    # Crimes over time (Year-Month Analysis)
    df['year_month'] = df['date_occ'].dt.to_period('M')
    crime_by_month = df['year_month'].value_counts().sort_index()
    plt.figure(figsize=(20, 15))
    crime_by_month.plot()
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Crimes')
    plt.title('Number of Crimes Over Time (Monthly Trend)')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    # Crime frequency by day of the week
    df['day_of_week'] = df['date_occ'].dt.day_name()
    plt.figure(figsize=(20, 15))
    sns.countplot(x='day_of_week', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Crimes')
    plt.title('Crime Frequency by Day of the Week')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    # Crime frequency by time of day
    df['time_occ'] = df['time_occ'].apply(lambda x: '{:04d}'.format(x))  # Convert to HHMM format
    df['hour_occ'] = df['time_occ'].str[:2].astype(int)
    plt.figure(figsize=(20, 15))
    sns.histplot(df['hour_occ'], bins=24, kde=True)
    plt.xlabel('Hour of Occurrence')
    plt.xticks(range(0, 24))
    plt.title('Crime Frequency by Time of Day')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

# Step 5: Comparative Analysis
def comparative_analysis(df):
    # Comparison of crime types over the years
    crime_type_year = df.groupby(['year', 'crm_cd_desc']).size().unstack(fill_value=0)
    plt.figure(figsize=(20, 15))
    crime_type_year.plot()
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.title('Comparison of Crime Types Over the Years')
    plt.xticks(crime_type_year.index, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

# Execute Analysis
if __name__ == "__main__":
    # Display basic information about the dataset
    basic_info(df)

    # Exploratory data analysis
    exploratory_analysis(df)

    # Temporal analysis of crime patterns
    temporal_analysis(df)

    # Comparative analysis of crime trends
    comparative_analysis(df)

    print("Analysis complete!")
