# Targeted-Marketing-Campaign-Optimization-Using-Customer-Segmentation

## Executive Summary

At **StyleSphere Inc.**, a leading online fashion retailer, understanding and leveraging customer behavior is pivotal to staying ahead in the competitive market. This project focuses on segmenting customers based on their spending scores to design a targeted marketing campaign. As a Marketing Analyst at StyleSphere, I analyzed a detailed dataset comprising demographic information and spending patterns of our diverse customer base to uncover actionable insights.

The analysis identified distinct customer segments, ranging from high-spending individuals to budget-conscious shoppers. This segmentation enables StyleSphere to align marketing initiatives with the unique preferences and behaviors of each group. For example:
- **High-Spending Customers**: Engaged through exclusive offers and premium product recommendations.
- **Budget-Conscious Shoppers**: Motivated by value-driven promotions and seasonal discounts.

By implementing a tailored marketing strategy informed by these insights, StyleSphere aims to:
- Increase customer engagement,
- Improve conversion rates, and
- Strengthen long-term customer loyalty.

Aligning campaigns with customer behavior not only optimizes outreach efforts but also positions StyleSphere as a customer-centric brand, enhancing its competitive edge in the online fashion market. This approach empowers the company to deliver personalized experiences, driving measurable improvements in sales and retention while reinforcing its reputation as a leader in the industry.

## Objectives

- **Customer Segmentation**: Analyze customer data to identify distinct segments based on spending scores and demographic factors, providing a foundation for understanding diverse customer behaviors and preferences at **StyleSphere Inc.**.
- **Targeted Marketing Strategy Development**: Design and implement personalized marketing strategies tailored to each identified segment, with a focus on delivering relevant messaging and promotions to drive engagement and increase sales.
- **Performance Measurement**: Define key performance indicators (KPIs) to assess the success of targeted campaigns, enabling continuous optimization based on customer response and engagement metrics.
- **Customer Retention Improvement**: Enhance customer loyalty by aligning strategies with the preferences of each segment, fostering long-term relationships and encouraging repeat purchases.
- **Sales Growth**: Boost revenue by aligning marketing efforts with spending patterns, ensuring campaigns maximize sales opportunities and deliver value to both high-spending and budget-conscious customers.

## Data Collection

- **Dataset Overview**:  
  The dataset used for this project, `mail_customers.csv`, was sourced from Kaggle and contains critical information about customers in the fashion retail sector. This dataset provides a robust foundation for segmenting customers and understanding their behavior.
  
- **Attributes**:  
  The dataset includes the following columns:  
  - `CustomerID`: Unique identifier for each customer, enabling precise tracking and segmentation.  
  - `Gender`: Indicates the gender of customers, offering insights into purchasing preferences.  
  - `Age`: Represents customer age, allowing for segmentation based on life stages and associated spending behaviors.  
  - `Annual Income (k$)`: Customer annual income in thousands of dollars, providing a measure of spending capacity.  
  - `Spending Score (1-100)`: A metric that quantifies customer engagement and purchasing behavior.

- **Data Source**:  
  The dataset was retrieved from Kaggle, ensuring a diverse and comprehensive representation of customer demographics and spending patterns. The quality and richness of this data make it ideal for driving actionable insights.

- **Preprocessing Steps**:  
  1. **Data Inspection**: Initial review to identify missing values, outliers, and data consistency issues.  
  2. **Data Cleaning**: Checked for duplicates and handling of missing data to ensure accuracy.  
  3. **Feature Engineering**: Creation of derived metrics where necessary to enhance segmentation analysis.  
  4. **Normalization**: Scaling numeric data to improve the performance of clustering algorithms.  

## Tools Used

This project leverages a range of Python libraries and tools that are essential for data analysis, visualization, and clustering. The tools include:

- **NumPy**: For numerical computations and efficient array manipulation.
- **Pandas**: For data cleaning, preprocessing, and exploratory data analysis.
- **Matplotlib**: For creating detailed and customizable visualizations.
- **Seaborn**: For generating aesthetically pleasing and informative plots to explore relationships within the data.
- **scikit-learn (KMeans)**: To implement clustering algorithms for customer segmentation.
- **Statsmodels**: For statistical modeling and evaluation where necessary.

### Why These Tools?

- These libraries are highly regarded in the data science community for their efficiency, reliability, and ease of use.
- Together, they enable seamless integration of data manipulation, visualization, and machine learning workflows, making them ideal for projects requiring end-to-end data analysis.

### Key Features of the Workflow

1. **Data Preprocessing**: Utilizing Pandas and NumPy to clean and transform raw data for further analysis.
2. **Exploratory Data Analysis (EDA)**: Leveraging Seaborn and Matplotlib to uncover patterns and relationships in the dataset.
3. **Clustering with KMeans**: Using scikit-learn's KMeans implementation to segment customers into meaningful groups.

This comprehensive toolset ensures that the project delivers actionable insights, paving the way for strategic decision-making.

# Exploratory Data Analysis

Exploratory Data Analysis (EDA) involves examining the dataset to uncover patterns, relationships, and insights. By utilizing visualizations like histograms and count plots, we can better understand customer demographics, spending habits, and income levels. This analysis plays a vital role in identifying distinct customer segments and aligning marketing strategies with data-driven insights.

## Gender Distribution

```python
plt.figure(figsize=(12,5))
sns.countplot(data=customer_data, x="Gender", palette="pastel")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
```
![Gender Distribution](https://github.com/user-attachments/assets/3020107e-3331-42a4-922b-1bc48537a5f1)


**Insights:**  
The count plot revealed the gender distribution within the dataset. It indicated a slightly higher number of female customers compared to male customers. This insight is critical for understanding the customer base and ensuring that marketing campaigns address the preferences of both genders effectively.

---

## Age Distribution

```python
plt.figure(figsize=(12,6))
sns.histplot(customer_data["Age"], kde=True, bins=20, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```
![Age Distribution](https://github.com/user-attachments/assets/e19793a5-0f50-42ed-88dc-16d615acdb0f)


**Insights:**  
The age distribution displayed a relatively uniform spread, with peaks between ages 30 and 40. Younger and middle-aged groups were the most represented, indicating a potential focus on these demographics. The kernel density estimate (KDE) curve emphasized this concentration, providing a smooth overview of customer age distribution.

---

## Annual Income Distribution

```python
plt.figure(figsize=(10,5))
sns.histplot(customer_data["Annual Income (k$)"], kde=True, bins=20, color="lightgreen")
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Frequency")
plt.show()
```
![Annual Income Distribution](https://github.com/user-attachments/assets/72c3a6e3-c47b-4b0a-94fa-ab7d74b84750)

**Insights:**  
The histogram showed that most customers have annual incomes ranging from $40k to $80k, with the highest frequency around $60k. The overlayed KDE curve confirmed the concentration in this range while highlighting the presence of a few high earners.


---

## Spending Score Distribution

```python
plt.figure(figsize=(12,6))
sns.histplot(customer_data["Spending Score (1-100)"], kde=True, bins=20, color="orange")
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Frequency")
plt.show()
```

![Spending Score Distribution](https://github.com/user-attachments/assets/dc197181-c6aa-4fc6-b7dd-5cd36b581db5)

**Insights:**  
The spending score distribution showed a concentration around mid-range scores (40-60) while also capturing significant numbers at both lower and higher ends. The KDE curve highlighted these trends, providing a clear representation of customer spending behavior.

## MODELLING

### MODELLING SELECTION

#### K-Means Clustering: Rationale for Selection

K-Means clustering was chosen for this customer segmentation project due to its simplicity and efficiency, making it well-suited for our dataset. The primary reasons for its selection include:

- **Objective Alignment**: K-Means effectively segments customers by minimizing intra-cluster variance, which aligns with our goal of grouping based on spending behavior and demographics.
  
- **Dataset Suitability**: The continuous numerical features (Age, Annual Income, Spending Score) fit well with K-Means requirements, and its computational efficiency is ideal for our moderate-sized dataset.

- **Interpretability**: K-Means provides clear cluster assignments that are easy to communicate to stakeholders.

- **Performance Comparison**: Compared to alternatives like Hierarchical Clustering and DBSCAN, K-Means showed better performance in terms of cluster quality and computational speed.

- **Practical Relevance**: Its widespread use in marketing analytics makes it a practical choice for StyleSphere, ensuring actionable insights.

Overall, K-Means strikes a balance between effectiveness and simplicity, making it the preferred method for this project.


## OPTIMAL NUMBER OF CLUSTERS

Determining the optimal number of clusters was a key component of our analysis, and we employed the elbow method for this purpose. This technique involved plotting the variance explained by each cluster against the number of clusters and identifying the point where the rate of improvement sharply decreased, forming an "elbow." 

By analyzing this plot, we were able to select a number of clusters that balanced complexity and interpretability, ensuring that each segment was distinct and meaningful.

### Code:
```python
# Elbow Method
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Elbow Curve Plot
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker="o", linestyle="-", color="b")
plt.title("Elbow Method for Optimal Cluster")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
```
![Elbow Method for Optimal Cluster](https://github.com/user-attachments/assets/a9591d3a-f37b-4e1f-9220-4ca8532b9eb5)


Based on the elbow method depicted in the graph, we observed that the inertia significantly decreased up to around 5 clusters, after which the rate of decline slowed considerably. This indicates that 5 clusters is likely the optimal choice, as it balances a clear segmentation of customers while avoiding unnecessary complexity.

## EVALUATING CLUSTERING PERFORMANCE

To assess the effectiveness of our clustering approach, we calculate metrics such as the silhouette score. This score helps evaluate the quality of the clusters by measuring how similar each data point is to its own cluster compared to other clusters. 

A higher silhouette score indicates better-defined clusters, providing insights into the effectiveness of our segmentation strategy.

### Code:
```python
from sklearn.metrics import silhouette_score

# Calculate the silhouette score
silhouette_avg = silhouette_score(scaled_features, customer_data["Cluster"])
print(f"Silhouette Score: {silhouette_avg:.2f}")
```
![Clustering Evaluation](https://github.com/user-attachments/assets/a3874582-26b5-47df-b508-43645dfdcfb5)

A silhouette score of `0.55` indicates a good level of cluster separation, suggesting that our clustering approach is effective. This score signifies that the data points are well-grouped within their respective clusters and sufficiently distant from points in other clusters. Overall, this result reflects a meaningful structure in our data, enhancing our confidence in the segmentation strategy used for targeted marketing efforts.

## FINDINGS

## Results: Customer Segmentation Findings

After performing K-Means clustering and validating the model, the analysis revealed **five distinct customer segments**. Each segment is characterized by unique spending patterns, demographic attributes, and purchasing behaviors. These findings will form the basis for our recommendations to optimize targeted marketing strategies at **StyleSphere Inc.**.

![Customer Segmentation](https://github.com/user-attachments/assets/5c211675-3e16-4c9e-8bf1-23f11140c578)

### Summary of Cluster Assignments

- **Segment 1 (High Spenders)**: Cluster 1
- **Segment 2 (Budget-Conscious Shoppers)**: Cluster 2
- **Segment 3 (Middle-Class Steady Spenders)**: Cluster 3
- **Segment 4 (Young Explorers)**: Cluster 4
- **Segment 5 (Low-Engagement Customers)**: Cluster 0

### Segment Characteristics

**High Spenders (Segment 1)**
- **Demographics**: Middle-aged individuals with high annual incomes.
- **Spending Score**: Very high, indicating frequent and significant purchases.

**Budget-Conscious Shoppers (Segment 2)**
- **Demographics**: Wide age range, lower annual incomes.
- **Spending Score**: Low, indicating infrequent or minimal purchases.

**Middle-Class Steady Spenders (Segment 3)**
- **Demographics**: Moderate annual incomes, balanced age distribution.
- **Spending Score**: Average.

**Young Explorers (Segment 4)**
- **Demographics**: Younger age group, moderate annual incomes.
- **Spending Score**: Above average, suggesting interest in trendy or fashionable products.

**Low-Engagement Customers (Segment 5)**
- **Demographics**: Diverse demographic attributes.
- **Spending Score**: Very low.

### Visualization of Customer Segments

```python
plt.figure(figsize=(10,5))
sns.scatterplot(
    x='Annual Income (k$)',
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    data=customer_data
)
plt.title("Customer Segments: Annual Income vs Spending Score")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
plt.show()
```

## Recommendation

The customer segmentation analysis has successfully identified distinct groups within the customer base at **StyleSphere Inc.**, providing valuable insights into spending behaviors and demographic patterns. These findings directly support the project's objectives of enhancing targeted marketing strategies, improving customer retention, and driving sales growth.

### Key Insights and Marketing Campaign Strategies

**High Spenders:**
- **Insight**: This segment consists of middle-aged individuals with high annual incomes and significant spending scores.
- **Strategy**: Develop premium offers and exclusive promotions tailored to their preferences, such as early access to new collections and personalized recommendations.

**Budget-Conscious Shoppers:**
- **Insight**: This group values affordability and often makes infrequent purchases.
- **Strategy**: Implement targeted promotions such as discounts, bundled offers, and loyalty rewards to encourage more frequent purchases and enhance overall engagement.

**Young Explorers:**
- **Insight**: Younger customers exhibit moderate spending scores and a keen interest in trendy products.
- **Strategy**: Launch trend-driven marketing campaigns through social media platforms, highlighting fashionable items and limited-time offers to capture their interest.

**Middle-Class Steady Spenders:**
- **Insight**: This segment shows balanced spending patterns across various demographics.
- **Strategy**: Promote seasonal collections and loyalty programs to foster repeat purchases and build long-term relationships.

**Low-Engagement Customers:**
- **Insight**: This diverse group demonstrates very low spending scores.
- **Strategy**: Conduct surveys to understand their disengagement reasons and address these through personalized re-engagement campaigns, offering incentives to rekindle their interest.

### Impact of Targeted Marketing Strategies

- **Improved Sales Performance**: Tailored marketing strategies are expected to increase conversion rates significantly by aligning promotional efforts with the preferences and behaviors of each segment.
  
- **Enhanced Customer Engagement**: Personalized outreach fosters deeper connections with customers, leading to improved satisfaction and loyalty.

- **Optimized Marketing Spend**: Focusing resources on high-value segments ensures efficient allocation of marketing budgets, maximizing return on investment (ROI).

- **Data-Driven Decisions**: The insights gained from clustering enable data-driven decision-making, reducing guesswork in marketing campaigns and improving overall effectiveness.

### Looking Ahead

Implementing the proposed strategies based on customer segmentation is anticipated to position **StyleSphere Inc.** as a customer-centric and innovative leader in the competitive fashion retail market. Regularly updating the segmentation model with new data will ensure that marketing efforts remain relevant and impactful, driving continuous growth and customer satisfaction.

## Limitations and Future Work

### Limitations

**Dataset Size and Scope:**  
The analysis is based on a single dataset, which may not fully capture the diversity of customer behavior across different geographic regions, product categories, or time periods. A larger and more diverse dataset would enhance the generalizability of the results.

**Static Data:**  
The dataset reflects customer behavior at a specific point in time. This static snapshot does not account for evolving customer preferences or market trends, which may impact the segmentation's long-term relevance.

**Limited Attributes:**  
Key behavioral and transactional variables, such as purchase frequency, product preferences, and marketing response rates, are missing. Incorporating such data would allow for more nuanced and actionable customer segments.

**Potential Biases:**  
The dataset may contain inherent biases, such as overrepresentation of certain demographics or spending patterns, which could skew the analysis and the resulting marketing strategies.

**Simplistic Clustering Algorithm:**  
While K-Means clustering is efficient and interpretable, it may not capture complex relationships in the data. More sophisticated methods like hierarchical clustering or DBSCAN could provide deeper insights into customer behavior.

### Future Work

**Incorporating Additional Data Sources:**  
Integrate data from multiple sources, such as online browsing history, product reviews, and past marketing campaign responses, to develop comprehensive customer profiles for segmentation.

**Exploring Advanced Clustering Techniques:**  
Utilize advanced clustering algorithms like Gaussian Mixture Models (GMM) or deep learning-based clustering to identify more intricate patterns and overlapping customer segments.

**Dynamic Segmentation:**  
Implement time-series analysis to track changes in customer behavior and spending patterns over time, allowing for dynamic and adaptive segmentation.

**Expanding Customer Attributes:**  
Include additional variables, such as geographic location, preferred communication channels, or brand loyalty scores, to refine the segmentation process and enhance strategy targeting.

**Experimentation and Validation:**  
Test the effectiveness of proposed marketing strategies through A/B testing and assess their impact on key performance indicators (KPIs) such as customer retention rates, revenue growth, and campaign ROI.
