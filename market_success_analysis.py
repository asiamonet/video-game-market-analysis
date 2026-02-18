# Import all required libraries
import pandas as pd #for data tables
import numpy as np #for math
from scipy import stats as st #for hypothesis testing
import matplotlib.pyplot as plt #for graphing/charts
import seaborn as sns #for statistical data visualization

# Load the dataset
df = pd.read_csv('/datasets/games.csv')
# Display basic information about the dataset
df.info()
"""
-The total number of records in the dataset is 16,715. -We have a mix of data types: float64 (6 columns) (year of release, NA, EU, JP Sales, Other sales & critic score). object (5 columns) (Name, Platform, Genre, User score, & Rating) -Obvious issues are the data type for User_score
should not be an object. There are missing values. Critic score having the highest number. Rating and User score also have significant gaps. Additionally, name and genre have 2 missing values each. Also, all of the column names have capitilization but standard Python practice is to use
lowercase. Also, Year of release has a wrong data type, it should be an integer. User score is an object, it should be a float for us to do math. -Ultimately, the fact that User_Score is an "object" while Critic_Score is a "float" suggests that users use "To Be Determined" for unreleased or low-
popularity games, while critics simply leave it blank.There are 2 rows where the Name and Genre are both missing. These might be corrupted rows we can just delete since 2 out of 16k is negligible.Every sales column is "Full" (16,715 records), but the scores are very "Empty." This
means we have plenty of financial data, but we lack feedback/reviews for nearly half the games in the "Ice" catalog.
"""
# Check for duplicate entries
print(df.duplicated().sum())
"""
No duplicate entries.
"""
# Convert column names to lowercase
df.columns= df.columns.str.lower()
# Verify the changes
print(df.columns)
# Check current data types
print(df.dtypes)
#Make changes to data types if necessary
#Describe the columns where the data types have been changed and why.

#Drop the missing years AND explicitly create a fresh copy
df = df.dropna(subset=['year_of_release']).copy()

#Change types without any copy warnings
df['year_of_release'] = df['year_of_release'].astype(int)
df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

#Check to make sure it's clean
print(df[['year_of_release', 'user_score']].dtypes)
"""
I changed year_or_release to int64 because years are discrete points in time. Since there were missing years (NaN), I removed those records to allow for a clean integer conversion. Additionally, I converted user_score to float64 because this column was originally an "object"
because it contained the string "TBD". Since we need to perform math calculations for our 2017 strategy, I converted these to numbers. Using errors='coerce' turned "TBD" into NaN (null), allowing the rest of the column to be treated as numeric decimals.
"""

"""
Because "TBD" is a string (text), it prevents us from performing any mathematical operations. By turning "TBD" into a null value, we can convert the entire column to a float. This allows us to
calculate statistics for the games that do have scores while ignoring the "TBD" entries in our math, rather than deleting those rows entirely and losing the sales data.
"""

# Examine missing values
print(df.isnull().sum())

# Calculate percentage of missing values
missing_percentages= (df.isna().sum()/len(df))*100
print('Percentage of missing values per column:')
print(missing_percentages.sort_values(ascending=False))

# Analyze patterns in missing values

# Calculating the overlap of missing scores
missing_critic = df['critic_score'].isna().sum()
missing_user = df['user_score'].isna().sum()
both_missing = (df['critic_score'].isna() & df['user_score'].isna()).sum()

print(f"Total missing critic scores: {missing_critic}")
print(f"Total missing user scores: {missing_user}")
print(f"Overlap (both missing): {both_missing}")

# Calculating the ratio to support the "evidence" requirement
overlap_pct = (both_missing / missing_user * 100)
print(f"Analysis: {overlap_pct:.1f}% of games without user scores also lack critic scores.")

"""
Over 50% of user_score and critic_score data is missing. While some may be due to 'TBD' conversions, much of it is likely due to the age of the games—many titles predate the digital era
of online reviews. Similarly, the 40% missing in rating likely stems from older games or regional titles (like those exclusive to Japan) that didn't undergo ESRB certification. The negligible
0.012% missing in name and genre are likely random technical errors or incomplete records from the original source.
"""

# Handle missing values based on analysis
# Your code here to handle missing values according to your strategy

#Drop name and genre missing values, only two rows are missing so we drop them
#useless for analysis
df= df.dropna(subset=['name','genre'])

#Fill rating with Unknown to keep them in our volume analysis.
df['rating']=df['rating'].fillna('Unknown')

#Final Check 
print('Missing values after cleanup:')
print(df.isna().sum())

"""
I have already explained in the previous markdown cell why I think values are missing. I dropped genre and name because they were technical errors and would not be useful in the analysis. I
also filled the missing values in rating with 'Unknown' to keep them in our volume analysis. I left both scores as NaN as i've already coerced tbd to NaN. I left them as is so I dont skew the analysis.
"""

# Calculate total sales across all regions and put them in a different column
df['total_sales']= df['na_sales']+ df['eu_sales']+ df['jp_sales']+df['other_sales']
print(df[['name', 'na_sales', 'jp_sales', 'eu_sales', 'other_sales', 'total_sales']].head())

# Create a DataFrame with game releases by year
games_by_year = df.groupby('year_of_release')['name'].count().sort_values(ascending=False).reset_index()

#rename columns for clarity 
games_by_year.columns= ['year_of_release', 'games_count']

print(games_by_year.head(10))

# Visualize the distribution of games across years
plt.figure(figsize=(12, 6))
plt.bar(games_by_year['year_of_release'], games_by_year['games_count'], color='skyblue', edgecolor='black')

#Formatting the chart
plt.title('Distribution of Video Game Releases (1980–2016)', fontsize=15)
plt.xlabel('Year of Release', fontsize=12)
plt.ylabel('Number of Games Released', fontsize=12)
plt.xticks(games_by_year['year_of_release'], rotation=45) # Rotate for readability
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

#Display the plot
plt.show()

# Display summary statistics for each year
# Detailed look at the release counts
print(games_by_year['games_count'].describe())

"""
I have observed that the most significant number of game releases occurred during the "golden era" between 2002 and 2011, where the industry consistently saw over 800 titles annually and
reached a massive peak of over 1,400 releases in 2008 and 2009. I noticed a clear pattern where the market rapidly expanded after the launch of the original PlayStation in the mid-90s, but then
began to contract sharply after 2011 as developers shifted toward more expensive, technically complex "AAA" titles and live-service models that require longer development cycles. Despite
this decline in the sheer quantity of games, I am confident that the data from 2014 to 2016 provides a sufficient foundation for 2017 predictions, as it captures the modern behavior of the
current console generation and the industry's pivot toward digital and mobile growth.
"""

# Calculate total sales by platform and year
platform_yearly_sales = df.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()
print(platform_yearly_sales.head(10))

# Create a heatmap of platform sales over time
platform_pivot = df.pivot_table(index='platform', 
                                columns='year_of_release', 
                                values='total_sales', 
                                aggfunc='sum').fillna(0)
#Plotting the heatmap
plt.figure(figsize=(16, 9))
sns.heatmap(platform_pivot, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Global Sales (Millions)'})

#Adding titles and labels
plt.title('Heatmap of Platform Sales Over Time', fontsize=18)
plt.xlabel('Year of Release', fontsize=14)
plt.ylabel('Platform', fontsize=14)
plt.show()

# Identify platforms with declining sales

# 1. Pivot data to get annual sales per platform using the full dataset
platform_growth = df.pivot_table(index='platform', columns='year_of_release', values='total_sales', aggfunc='sum').fillna(0)

# 2. Calculate the change between the two most recent years available in the data
last_year = platform_growth.columns[-1]
prev_year = platform_growth.columns[-2]

platform_growth['absolute_change'] = platform_growth[last_year] - platform_growth[prev_year]
platform_growth['percent_change'] = (platform_growth['absolute_change'] / platform_growth[prev_year] * 100).replace([np.inf, -np.inf], 0)

# 3. Filter for declining platforms and sort by sharpest decline
declining_platforms = platform_growth[platform_growth['absolute_change'] < 0][['absolute_change', 'percent_change']].sort_values(by='absolute_change')

print(f"Platforms with Declining Sales ({prev_year} to {last_year}):")
print(declining_platforms)

"""
Significant Decline: I can see that the Xbox 360, PlayStation 3, and Wii are in a sharp downward spiral by 2016. These were the "Kings" of the previous generation, but their sales have
plummeted as the audience migrated to newer hardware. Even the 3DS is showing a steady decline, despite being a strong performer for years.

Vanished Platforms: I noticed that legendary names like the DS, PS2, and GameCube have completely disappeared from the market (zero sales) by 2014-2016. In fact, most platforms in
this dataset have a very "hard" end date where sales simply stop.

The 10-Year Lifecycle: I’ve calculated that the typical "life" of a console is approximately 10 years. It usually takes 3 to 5 years to reach the absolute peak of popularity, and then another 5
years to fade into obsolescence as the next generation takes over.

The "Outlier": Consistent Sales

Interestingly, the PC is the only platform that shows "consistency." While it doesn't always have the highest peaks, it doesn't "die" like a console does. Because the hardware is constantly being
upgraded, it survives through every generation, whereas consoles like the PS2 are eventually replaced by the PS3.
"""

# Filtering the dataset to include only the most relevant years
# I'm choosing 2013 as the start because it's the birth of the current leaders
df_relevant = df[df['year_of_release'] >= 2013].copy()

# Quick check to see how many records we have left
print(f"Number of games in our relevant period: {len(df_relevant)}")
print(df_relevant['year_of_release'].unique())

"""
I’ve decided that for our final report, we should focus exclusively on the period from 2013 to 2016. I believe this is the most honest way to look at the data if we want our 2017 strategy to
actually succeed.

My Reasoning for the Decision:
I chose the year 2013 as our starting point because it represents a "reset" for the entire industry. I noticed in my earlier analysis that this was the launch year for both the PlayStation 4 and the
Xbox One. By 2013, the previous dominant consoles (PS3 and Xbox 360) had already passed their prime and were beginning to fade. If I included data from 2010 or 2011, I’d be looking at a
world where the Nintendo Wii and DS were still kings—but that world no longer exists in 2017.

Reflecting Current Market Conditions:
This 2013–2016 period reflects the "Modern Era" of gaming. I see three specific factors in this data that match the market conditions for 2017:

Hardware Dominance: This period captures the full growth curve of the PS4 and Xbox One, which are the platforms our "Ice" store will rely on most.

The Shift in Volume: As I noted before, the number of releases dropped after 2012. This period reflects the current "quality over quantity" market where fewer, larger games dominate the sales
charts.

The PC Exception: By looking at these years, I can see that the PC remains a stable, consistent earner even while other consoles rise and fall, which is a key trend that continues into 2017.

Factors Influencing My Decision:
My decision was primarily driven by the 10-year platform lifecycle I discovered. Since a console usually only has about 5 years of "peak" relevance, data from 2007 is effectively "dead" data. I
also wanted to ensure that our Total Sales calculations weren't being skewed by "phantom" sales from platforms that are no longer being manufactured. By cutting the data at 2013, I am
ensuring that every dollar we analyze is coming from a consumer behavior that is still relevant today.
"""

#Using my selected time period, analyze platform performance:
# Analyze platform sales trends
# I'm grouping our relevant data by year and platform to see the trends
platform_trends = df_relevant.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

# I'll pivot this to make it easier to read as a time-series
platform_trends_pivot = platform_trends.pivot(index='year_of_release', columns='platform', values='total_sales').fillna(0)

print(platform_trends_pivot)

# Sort platforms by total sales

# I am summing up the sales for our relevant period and sorting them
platform_sales_summary = df_relevant.groupby('platform')['total_sales'].sum().sort_values(ascending=False).reset_index()

# Let's see the ranking
print("Top Platforms by Sales (2013-2016):")
print(platform_sales_summary)

# Visualize top platforms
# Calculate year-over-year growth for each platform
# Your code here to calculate and visualize platform growth rates

# 1. Ranking the Top Platforms in the Modern Era
top_sales = df_relevant.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

# 2. Calculating Year-over-Year Growth Rate
# I pivot the data first to get years as rows and platforms as columns
pivot_sales = df_relevant.pivot_table(index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')
growth_rates = pivot_sales.pct_change() * 100  # Multiplying by 100 to get a percentage

# 3. Visualizing the Growth
plt.figure(figsize=(12, 6))
for platform in growth_rates.columns:
    # I'm only plotting the most relevant platforms to keep the chart clean
    if platform in top_sales.head(5).index:
        plt.plot(growth_rates.index, growth_rates[platform], marker='o', label=platform)

plt.title('Year-over-Year Sales Growth Rate (%)', fontsize=15)
plt.ylabel('Growth Rate (%)', fontsize=12)
plt.axhline(0, color='black', linestyle='--') # The "Zero Line" shows where growth stops
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Create box plot of sales by platform
# 1. First, I'll look at the full distribution (including the giant hits)
plt.figure(figsize=(14, 7))
sns.boxplot(x='platform', y='total_sales', data=df_relevant)

plt.title('Global Sales Distribution by Platform (2013-2016)')
plt.xlabel('Platform')
plt.ylabel('Global Sales (Millions)')
plt.show()

# Calculate detailed statistics for each platform

# 1. Group by platform and grab the total_sales column
# 2. Use describe() to get count, mean, std, min, 25%, 50%, 75%, and max
platform_stats = df_relevant.groupby('platform')['total_sales'].describe()

# 3. Sort by 'count' or 'mean' to see the leaders at the top
platform_stats = platform_stats.sort_values(by='mean', ascending=False)

# Display the result
print(platform_stats)

"""
My analysis of the gaming market between 2013 and 2016 reveals a highly concentrated industry defined by rapid generational transitions and a heavy reliance on "blockbuster" hits. We found
that the typical lifecycle of a console lasts approximately ten years, and by 2016, the market had firmly shifted away from legacy hardware like the PlayStation 3 and Xbox 360 toward the
PlayStation 4 and Xbox One. While the PlayStation 4 emerged as the undisputed leader in both total revenue and volume of titles, the statistical data shows a significant gap between average
and median sales across all major platforms. This disparity indicates that the vast majority of revenue is generated by a small percentage of high-performing "superstar" titles, while the
typical game sells much more modestly. Furthermore, the PC remains the only platform characterized by long-term stability rather than a boom-and-bust cycle, though its individual'
game sales tend to be lower on average than their console counterparts. Ultimately, for a successful 2017 strategy, investment should be prioritized toward the PlayStation 4 and Xbox
One ecosystems, as they currently hold the highest momentum and the greatest potential for high-revenue outliers.
"""

# Choose a popular platform based on previous analysis. Analyze how reviews affect sales.
# Defining the chosen popular platform based on the sales and life cycle analysis
chosen_platform = 'PS4'

# Creating a filtered subset for this platform to use in subsequent analysis
platform_data = df_relevant[df_relevant['platform'] == chosen_platform]

print(f"Selected platform for detailed analysis: {chosen_platform}")
print(f"Total records for {chosen_platform} in relevant period: {len(platform_data)}")

"""
Based on my previous analysis, I have selected the PlayStation 4 (PS4) as our focus. It is the
undisputed market leader for our relevant period (2013–2016), so understanding what drives its
sales is the most valuable insight we can get for our 2017 strategy.
"""

# Create scatter plots for both critic and user scores
# Filtering data specifically for PS4 and dropping missing values
ps4_subset = df_relevant[df_relevant['platform'] == 'PS4'].dropna(subset=['critic_score', 'user_score'])

# 1. Critic Score vs. Total Sales
plt.figure(figsize=(10, 5))
sns.scatterplot(data=ps4_subset, x='critic_score', y='total_sales', color='royalblue', alpha=0.7)
plt.title('PS4: Critic Score vs. Global Sales (2013-2016)')
plt.xlabel('Critic Score')
plt.ylabel('Global Sales (Millions)')
plt.show()

# 2. User Score vs. Total Sales
plt.figure(figsize=(10, 5))
sns.scatterplot(data=ps4_subset, x='user_score', y='total_sales', color='darkorange', alpha=0.7)
plt.title('PS4: User Score vs. Global Sales (2013-2016)')
plt.xlabel('User Score')
plt.ylabel('Global Sales (Millions)')
plt.show()

# Critic Scores
# User Scores
# Calculate correlations

# Calculating the correlation matrix for the PS4
# 1.0 = Perfect positive relationship
# 0.0 = No relationship at all
correlation_results = ps4_subset[['critic_score', 'user_score', 'total_sales']].corr()

print("Correlation Coefficients for PS4:")
print(correlation_results['total_sales'])

"""
Why is there such a big difference?

The moderate positive correlation found between critic scores and sales, contrasted with the near-zero correlation for user scores, highlights a fundamental difference in how these ratings
interact with the market. Professional reviews typically carry more weight because they are released prior to a game's launch, directly fueling the pre-order momentum and the massive
"Day 1" sales spikes that define a blockbuster's success. In contrast, user scores often accumulate long after the initial purchasing decisions have been made, making them a trailing
indicator rather than a predictive one. Furthermore, user ratings are frequently susceptible to "review bombing," where players may assign a game a zero over technical glitches or narrative
controversies that do not necessarily deter the general public from buying the title. Consequently, while professional critical consensus serves as a reliable barometer for
commercial potential, user sentiment remains a disconnected metric that reflects player experience rather than market reach.
"""

#Comparing sales performance of games across different platforms:
# Find games released on multiple platforms
# 1. Count how many platforms each game title appears on
game_counts = df_relevant.groupby('name')['platform'].nunique()

# 2. Filter for names that appear on more than one platform
multiplatform_names = game_counts[game_counts > 1].index

# 3. Create a dataframe containing only these cross-platform titles
df_multi = df_relevant[df_relevant['name'].isin(multiplatform_names)]

# Let's see how the sales compare for a few famous examples
example_games = df_multi.groupby(['name', 'platform'])['total_sales'].sum().unstack().fillna(0)
print(example_games.head(10))

"""
Conclusion of Findings:
Based on the cross-platform comparison of shared titles, it is evident that sales distribution heavily favors the more modern consoles, reinforcing our identification of the PlayStation 4 and
Xbox One as the market leaders for the 2013–2016 period. While older platforms like the PlayStation 3 and Xbox 360 still capture a portion of the market for multi-platform releases—as
seen in the sales for titles like 2014 FIFA World Cup Brazil—their share is significantly smaller compared to their successors. Furthermore, specialized handheld platforms like the 3DS and PS
Vita show a distinct niche dominance for specific titles, such as Frozen: Olaf's Quest or the Super Robot Wars series, which highlights that some software is culturally or hardware-specific.
Ultimately, the data confirms that for high-revenue, multi-platform games, the PlayStation 4 consistently captures the highest total sales volume, aligning with our broader finding that it is
the most profitable platform for the "Ice" store to prioritize in its 2017 inventory strategy.
"""

# Compare sales across platforms for these games
# Your code here to analyze and visualize cross-platform performance

# 1. Identify games that exist on more than one platform
multiplatform_games = df_relevant.groupby('name').filter(lambda x: x['platform'].nunique() > 1)

# 2. To keep the chart readable, I'll pick the top 5 multiplatform games by total revenue
top_titles = multiplatform_games.groupby('name')['total_sales'].sum().sort_values(ascending=False).head(5).index
df_comparison = multiplatform_games[multiplatform_games['name'].isin(top_titles)]

# 3. Create a pivot table for the grouped bar chart
comparison_pivot = df_comparison.pivot_table(index='name', columns='platform', values='total_sales', aggfunc='sum')

# 4. Plotting the side-by-side comparison
comparison_pivot.plot(kind='bar', figsize=(14, 7), width=0.8)
plt.title('Sales Comparison Across Platforms (2013-2016)')
plt.ylabel('Total Sales (Millions)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Analyze genre performance

#Calculate total sales and count of games per genre
genre_analysis = df_relevant.groupby('genre')['total_sales'].agg(['sum', 'count', 'mean']).sort_values(by='sum', ascending=False)
plt.figure(figsize=(12, 6))
genre_analysis['sum'].plot(kind='bar', color='teal', edgecolor='black')
plt.title('Total Global Sales by Genre (2013-2016)')
plt.ylabel('Total Sales (Millions)')
plt.xlabel('Genre')
plt.show()

#Check the average sales per game to find the "most profitable" genre
print(genre_analysis.sort_values(by='mean', ascending=False))

# Sort genres by total sales
# 1. Grouping the filtered data by genre and summing the total sales
genre_sales = df_relevant.groupby('genre')['total_sales'].sum().sort_values(ascending=False).reset_index()

# 2. Displaying the sorted results
print("Genres Ranked by Total Sales (2013-2016):")
print(genre_sales)

# Visualize genre distribution

# 1. Preparing the data: Sum of sales and average sales per genre
genre_stats = df_relevant.groupby('genre')['total_sales'].agg(['sum', 'mean']).sort_values(by='sum', ascending=False)

# 2. Plotting Total Sales by Genre
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_stats.index, y=genre_stats['sum'], palette='viridis')

plt.title('Total Global Sales by Genre (2013-2016)', fontsize=15)
plt.ylabel('Total Sales (Millions)', fontsize=12)
plt.xlabel('Genre', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Calculate market share for each genre

# 1. Calculate the grand total of all sales in this period
total_market_sales = df_relevant['total_sales'].sum()

# 2. Calculate the percentage share for each genre
genre_share = (df_relevant.groupby('genre')['total_sales'].sum() / total_market_sales * 100).sort_values(ascending=False)

# 3. Display the market share
print("Market Share by Genre (%):")
print(genre_share.map('{:.2f}%'.format))

"""
Genre Analysis:
My analysis of the gaming landscape from 2013 to 2016 reveals a market where a few dominant genres command the vast majority of consumer spending. Action and Shooter games
consistently lead the charts in total revenue, forming the backbone of the industry's financial health during this period. While the Action genre boasts the highest number of individual titles
and the largest overall market share, the Shooter genre often proves to be more efficient, generating massive returns with a smaller, more concentrated pool of blockbuster releases.
Sports and Role-Playing games maintain a stable and significant presence in the middle of the market, benefiting from established franchises and loyal fanbases that ensure consistent year-
over-year performance.

When examining recent momentum, we see a clear decline in specialized or "niche" categories such as Puzzle, Strategy, and Adventure games. These genres have struggled to maintain
relevance in the modern era, capturing only a tiny fraction of the total market share compared to high-intensity, cinematic experiences. Conversely, the Shooter genre has shown remarkable
resilience; even as older platforms fade away, Shooters continue to drive hardware adoption on the PlayStation 4 and Xbox One. This suggests that while total game counts for some genres
may fluctuate, the consumer appetite for large-scale, competitive experiences is a primary growth driver as we look toward 2017.

The most striking discovery in this analysis is the disparity in average performance across different genres. Although Action games generate the most total money, their average sales per
game are often lower than those of Shooters or Sports titles because the market is saturated with many smaller, less successful projects. In contrast, the Shooter genre consistently delivers
the highest average revenue per title, indicating that a "typical" shooter is a much more powerful financial asset than a typical adventure or strategy game. For a retailer like Ice, this means that
while Action games provide variety, Shooters represent the highest-value inventory, as they are statistically more likely to become the multi-million-selling outliers that define a successful year.
"""

"""
In this section, I will analyze the gaming market characteristics across three major regions:
North America (NA), Europe (EU), and Japan (JP). My analysis will focus on platform
preferences, genre popularity, and the impact of ESRB ratings in each region.
"""

#examining platform performance across different regions:
# Function to analyze platform performance by region

def analyze_region_platforms(df, region_sales_column):
    """
    Groups data by platform, calculates total sales for a specific region,
    and identifies the top 5 platforms with their market shares.
    """
    # Grouping by platform and summing the specific regional sales
    region_data = df.groupby('platform')[region_sales_column].sum().sort_values(ascending=False)
    
    # Calculating the top 5 platforms for the region
    top_5 = region_data.head(5)
    
    # Calculating the percentage of market share within the region
    total_region_sales = region_data.sum()
    shares = (top_5 / total_region_sales) * 100
    
    return top_5, shares

# Analyze each region

# 1. Define the regions and their corresponding columns
regions = [('na_sales', 'North America'), ('eu_sales', 'Europe'), ('jp_sales', 'Japan')]

# 2. Iterate through each region and call our function
for col, name in regions:
    top_5, shares = analyze_region_platforms(df_relevant, col)
    
    print(f"--- {name} ---")
    # Combining results into a small table for easy reading
    results = pd.DataFrame({
        'Total Sales (M)': top_5,
        'Market Share (%)': shares
    })
    print(results)
    print("\n")

#create a comparative analysis of platform performance across regions
# 1. Summarize all regions into one dataframe
comparative_platforms = df_relevant.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales']].sum()

# 2. Sort by NA sales so the bars follow a logical descending order
comparative_platforms = comparative_platforms.sort_values(by='na_sales', ascending=False)

# Visualize cross-regional comparison for top platforms

#Plotting the comparison
comparative_platforms.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])

plt.title('Regional Composition of Platform Sales (2013-2016)', fontsize=15)
plt.ylabel('Total Sales (Millions)')
plt.xlabel('Platform')
plt.legend(['North America', 'Europe', 'Japan'], title='Region')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# examine genre preferences across regions:
# Function to analyze genre performance by region
def analyze_region_genres(df, region_sales_column):
    """
    Groups data by genre, calculates total sales for a specific region, 
    and returns the top 5 genres with their market shares.
    """
    # Grouping by genre and summing sales for the specific region
    genre_data = df.groupby('genre')[region_sales_column].sum().sort_values(ascending=False)
    
    # Selecting the top 5 genres
    top_5 = genre_data.head(5)
    
    # FIX: Use genre_data.sum() to get the total for the entire region
    total_region_sales = genre_data.sum() 
    shares = (top_5 / total_region_sales) * 100
    
    return top_5, shares

# Calling the function for NA, EU, and JP
regions = [('na_sales', 'North America'), ('eu_sales', 'Europe'), ('jp_sales', 'Japan')]

for col, name in regions:
    top_genres, shares = analyze_region_genres(df_relevant, col)
    print(f"--- Top 5 Genres in {name} ---")
    results = pd.DataFrame({
        'Total Sales (M)': top_genres,
        'Market Share (%)': shares.round(2)
    })
    print(results)
    print("\n")

#compare genre preferences across regions:
# Create a comparative genre analysis

# 1. Aggregate regional sales by genre
comparative_genres = df_relevant.groupby('genre')[['na_sales', 'eu_sales', 'jp_sales']].sum()

# 2. Sort by North American sales to maintain a logical order
comparative_genres = comparative_genres.sort_values(by='na_sales', ascending=False)

# 3. Plotting the side-by-side comparison
comparative_genres.plot(kind='bar', figsize=(14, 7), width=0.8)

plt.title('Genre Sales Comparison: NA vs EU vs JP (2013-2016)')
plt.ylabel('Sales (Millions)')
plt.xlabel('Genre')
plt.legend(['North America', 'Europe', 'Japan'], title='Region')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Displaying the raw numbers for precision
print(comparative_genres)

"""
Interpretation of Regional Differences:
The regional data reveals a high degree of similarity between the North American and European markets, where Action, Shooter, and Sports genres dominate the top three spots. In these
Western territories, Shooters are a massive revenue driver, nearly rivaling Action games in scale. However, the Japanese market presents a significant cultural divergence; it is the only region
where Role-Playing Games (RPGs) claim the top position, outperforming Action games and dwarfing the sales of Shooters, which barely make the top five. Additionally, Europe shows a
unique preference for Racing titles, which displace the "Misc" category found in the North American top five. For the "Ice" store's 2017 strategy, this suggests that inventory should be
heavily localized, prioritizing high-octane Shooters for Western audiences while focusing on a robust RPG catalog for the Japanese market.
"""

#examine how ESRB ratings affect sales in each region:
# Function to analyze ESRB rating impact
def analyze_region_esrb(df, region_sales_column):
    """
    Groups data by ESRB rating, calculates total sales for a specific region,
    and returns the breakdown to see which content levels perform best.
    """
    # Filling missing ratings with 'Unknown' to capture unrated (often JP) titles
    df_copy = df.copy()
    df_copy['rating'] = df_copy['rating'].fillna('Unknown')
    
    # Grouping by rating and summing the regional sales
    esrb_data = df_copy.groupby('rating')[region_sales_column].sum().sort_values(ascending=False)
    
    # Calculating market share within the region
    total_region_sales = esrb_data.sum()
    shares = (esrb_data / total_region_sales) * 100

    return esrb_data, shares

# Analyze ESRB impact for each region

# 1. Define the regions and their sales columns
regions = [('na_sales', 'North America'), ('eu_sales', 'Europe'), ('jp_sales', 'Japan')]

# 2. Iterate and print the breakdown for each
for col, name in regions:
    esrb_sales, esrb_shares = analyze_region_esrb(df_relevant, col)
    print(f"--- {name} ---")
    results = pd.DataFrame({
        'Sales (M)': esrb_sales,
        'Market Share (%)': esrb_shares
    })
    print(results)
    print("\n")

"""
Interpretation of Regional Patterns

The ESRB analysis reveals a sharp contrast between Western markets and Japan, specifically regarding the influence of "Mature" (M) rated content and the presence of unrated titles. In
North America and Europe, M-rated games are the clear market leaders, accounting for over 37% of total sales in both regions. This indicates that the Western audience is heavily driven by
titles designed for older players, such as mainstream shooters and action-adventure epics. While "Everyone" (E) rated games maintain a solid second place in the West, the overall strategy for
these regions should clearly prioritize the distribution of high-profile, adult-oriented titles to maximize revenue.

Japan’s data presents an entirely different picture, with the "Unknown" category dominating the market at a staggering 60%. This massive gap exists because the ESRB is a North American
body; most games developed specifically for the Japanese domestic market are rated by CERO, Japan’s local rating board, rather than the ESRB. Consequently, while the ESRB data suggests
that "Teen" (T) and "Everyone" (E) ratings are popular, the high volume of "Unknown" titles means that ESRB ratings are not a reliable metric for the Japanese market. For a global strategy
in 2017, the "Ice" store should rely heavily on ESRB ratings for inventory planning in the West, but must shift to local Japanese trends or CERO classifications when stocking for the Japanese region.
"""

#Hypothesis Testing: 
#Hypothesis 1: Xbox One vs. PC User Ratings

# 1. Filter the data for each platform
xone_ratings = df_relevant[(df_relevant['platform'] == 'XOne') & (df_relevant['user_score'].notna())]['user_score']
pc_ratings = df_relevant[(df_relevant['platform'] == 'PC') & (df_relevant['user_score'].notna())]['user_score']

# 2. Perform the t-test
results_platforms = st.ttest_ind(xone_ratings, pc_ratings, equal_var=False)

print(f"p-value (Platforms): {results_platforms.pvalue}")

if results_platforms.pvalue < 0.05:
    print("We reject the null hypothesis: User ratings differ between Xbox One and PC.")
else:
    print("We cannot reject the null hypothesis: Average user ratings are likely the same.")

#Hypothesis 2: Action vs. Sports User Ratings
# 1. Filter the data for each genre
action_ratings = df_relevant[(df_relevant['genre'] == 'Action') & (df_relevant['user_score'].notna())]['user_score']
sports_ratings = df_relevant[(df_relevant['genre'] == 'Sports') & (df_relevant['user_score'].notna())]['user_score']

# 2. Perform the t-test
results_genres = st.ttest_ind(action_ratings, sports_ratings, equal_var=False)

print(f"p-value (Genres): {results_genres.pvalue}")

if results_genres.pvalue < 0.05:
    print("We reject the null hypothesis: User ratings for Action and Sports are different.")
else:
    print("We cannot reject the null hypothesis: There is no significant difference in ratings.")

"""
The formulation of our statistical hypotheses follows the fundamental principle that the null hypothesis must always be expressed as a statement of equality, representing the assumption
that there is no significant difference between the groups being compared. For the first test, the null hypothesis posits that the average user ratings for the Xbox One and PC platforms are
identical, while the alternative hypothesis suggests that a statistically significant difference exists between them. Similarly, for the second test, the null hypothesis assumes that the average
user ratings for the Action and Sports genres are equal, with the alternative hypothesis stating that these ratings are actually different. By framing the alternative hypotheses as "different"
rather than specifying one is "higher" or "lower," I have employed a two-tailed test, which is a more rigorous approach that accounts for variations in either direction.

The choice of the t-test for independent samples as our testing criterion is based on the nature of the data and the structure of the groups. Because we are comparing the means of two
independent populations—where the rating of a game on one platform or in one genre does not influence the rating of another—this test is the most appropriate statistical tool. I specifically
utilized Welch’s t-test by setting the variance parameter to false, which allows the test to remain accurate even if the groups have different sample sizes or different levels of spread in their data.
This ensures that the results are not biased by the fact that we might have far more Action game reviews than Sports game reviews in our dataset.

To determine whether the results were statistically significant, I applied an alpha threshold of 0.05, which serves as the standard benchmark for business and social science research. This
threshold means that we only reject the null hypothesis if there is less than a 5% probability that the observed difference in user ratings occurred due to random chance. If the resulting p-value
is lower than this 0.05 threshold, we conclude that the difference is real and not just a fluke of the data. This level of scrutiny provides the "Ice" store with a reliable foundation for making
inventory decisions, as it distinguishes between minor, accidental fluctuations and genuine shifts in consumer sentiment.
"""

"""
General Conclusion

My comprehensive analysis of the video game market from 2013 to 2016 provides a data-driven roadmap for the "Ice" store's 2017 strategy. The transition between console generations is now
complete, with the PlayStation 4 emerging as the most stable and profitable global platform. While the Xbox One remains a critical competitor in North America, its performance in Europe
and Japan is significantly weaker, suggesting that inventory for the Xbox should be prioritized primarily for Western markets. Conversely, the Nintendo 3DS remains the dominant force in
Japan, highlighting a unique regional preference for handheld gaming that requires a specialized stock approach compared to the console-heavy markets of the West.

The study of genre performance confirms that Action and Shooter games are the primary drivers of total revenue, though they operate under different financial dynamics. Action games rely on a
high volume of releases to maintain market share, whereas Shooters benefit from extremely high average sales per title, making them the most efficient "hit-makers" for a retail inventory.
However, these trends shift dramatically in Japan, where Role-Playing Games frequently outperform the Shooter genre. This regional divergence indicates that a successful 2017
campaign must be localized; Western stores should focus on high-intensity "Mature" rated Shooters and Action titles, while the Japanese market requires a heavy emphasis on RPGs and
handheld-accessible content.

Statistical hypothesis testing reinforced the idea that platform choice alone does not dictate user satisfaction, as we found no significant difference in average user ratings between Xbox One and
PC. However, the distinct difference in ratings between Action and Sports genres suggests that different types of players hold their games to different standards, with the Sports audience often
being more critical of their annual releases. Finally, the impact of ESRB ratings proved significant in the West, where "Mature" rated games dominate the top of the sales charts. Moving into 2017,
the most successful strategy will involve aggressive promotion of PS4 blockbusters in the Action and Shooter genres for the global market, while maintaining a niche, high-value handheld and
RPG focus specifically for Japanese consumers.
"""

