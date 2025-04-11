# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#TE MAAKE
#Tennis player's performance 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor


sport_perf = pd.read_csv("C:/Projects/sport_perform.csv")
tennis = sport_perf[sport_perf.sport=="Tennis"]

#Corr matrix
tennis2 = tennis.drop(columns=['Unnamed: 0','type','sport','gender'])
correlation_matrix=tennis2.corr()
print(correlation_matrix)

sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.show()

Corr2 = correlation_matrix[abs(correlation_matrix.perform)>=0.4]
sns.heatmap(Corr2,annot=True,cmap='coolwarm')
plt.show()


#Graphs
conditions_x12 = [(tennis.x12 < 35),(tennis.x12 >= 35) & 
              (tennis.x12 < 45),(tennis.x12 >= 45)]
choices = ['0-Minimum','1-Moderate','2-Maximum']
#Create the categorical variable_x12
tennis['cart_x12'] = np.select(conditions_x12, choices)
sorted_categories = sorted(tennis['cart_x12'].unique())

sns.boxplot(x='cart_x12', y='perform', data=tennis,
            order=sorted_categories, palette="rocket_r",
            notch=True, width=0.5)
plt.title("Perfomance by level of equipment used")
plt.xlabel("Equipment level")
plt.ylabel("Perfomance")
plt.show()


conditions_x25 = [(tennis.x25 < 35),(tennis.x25 >= 35) & 
              (tennis.x25 < 45),(tennis.x25 >= 45)]
choices1 = ['0-Low','1-Medium','2-High']

#Create the categorical variable_x25
tennis['cart_x25'] = np.select(conditions_x25, choices1)
sorted_categories = sorted(tennis['cart_x25'].unique())

sns.boxplot(x='cart_x25', y='perform', data=tennis,
            order=sorted_categories, palette="YlOrBr",
            notch=True, width=0.5)
plt.title("Perfomance by level of tactical skills")
plt.xlabel("Tactical skills level")
plt.ylabel("Perfomance")
plt.show()

#x26
plt.hexbin(tennis["x26"], tennis["perform"], gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count')
plt.title("Hexbin Plot performance vs fitness")
plt.xlabel("Fitness")
plt.ylabel("Performance")
plt.show()
###################################################################################################
# Categorize performance into three levels: Low, Medium, High
bins = [0, 50, 75, 100]
labels = ['Low', 'Medium', 'High']
tennis['perform_cat'] = pd.cut(tennis['perform'], bins=bins, labels=labels)

plt.figure(figsize=(10, 6))
sns.histplot(data=tennis, x='x24', hue='perform_cat', multiple='stack', bins=20, palette=['springgreen', 'royalblue', 'orange'])
plt.title('Distribution of Speed and Velocity by Performance Levels in Tennis')
plt.xlabel('Speed and Velocity')
plt.ylabel('Frequency')
plt.legend(title='Performance Levels', loc='upper right', labels=['Low Performers (Below 50)', 
                                                                      'Medium Performers (50-75)', 
                                                                      'High Performers (Above 75)'])
plt.show()


#x12
plt.figure(figsize=(10, 6))
sns.histplot(data=tennis, x='x12', hue='perform_cat', multiple='stack', bins=20, palette=['olive', 'cyan', 'yellow'])
plt.title("Perfomance by level of equipment used")
plt.xlabel("Equipment level")
plt.ylabel("Perfomance")
plt.legend(title='Performance Levels', loc='upper right', labels=['Low Performers (Below 50)', 
                                                                      'Medium Performers (50-75)', 
                                                                      'High Performers (Above 75)'])
plt.show()

#x25
plt.figure(figsize=(10, 6))
sns.histplot(data=tennis, x='x25', hue='perform_cat', multiple='stack', bins=20, palette=['lime', 'cornflowerblue', 'orange'])
plt.title("Perfomance by level of tactical skills")
plt.xlabel("Tactical skills level")
plt.ylabel("Perfomance")
plt.legend(title='Performance Levels', loc='upper right', labels=['Low Performers (Below 50)', 
                                                                      'Medium Performers (50-75)', 
                                                                      'High Performers (Above 75)'])
plt.show()

#x18
plt.figure(figsize=(10, 6))
sns.histplot(data=tennis, x='x18', hue='perform_cat', multiple='stack', bins=20, palette=['rebeccapurple', 'paleturquoise', 'lightcoral'])
plt.title('Distribution of Mental Toughness and Focus by Performance Levels in Tennis')
plt.xlabel('Mental Toughness and Focus')
plt.ylabel('Frequency')
plt.legend(title='Performance Levels', loc='upper right', labels=['Low Performers (Below 50)', 
                                                                      'Medium Performers (50-75)', 
                                                                      'High Performers (Above 75)'])
plt.show()
################################################################################################################################################
##############################################################################################################################

#Phase 2 of the project
# Categorize Performance and Psychological Factors (x23)
tennis['perform_cat'] = pd.cut(tennis['perform'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])
tennis['x23_cat'] = pd.cut(tennis['x23'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])

# Cross-tab to explore relationships between performance and psychological factors
tennis3=pd.crosstab(tennis['perform_cat'], tennis['x23_cat'])

# Chi-Square Test for Independence between performance and psychological factors

#H0: perfomance and psychological factors are independent vs Ha:perfomance and psychological factors are dependent
ans = chi2_contingency(tennis3)
pval=ans.pvalue
alpha= 0.05
if pval < alpha:
    print("Reject H0, perfomance and psychological factors are dependent")
else:
    print("Do not reject H0,perfomance and psychological factors are independent")
#Do not reject H0,perfomance and psychological factors are independent

# Independent t-test for gender differences in performance
#H0: There is no difference in performance between males and females.
#Ha: There is a difference in performance between males and females.
male_performance = tennis[tennis['gender'] == 'Male']['perform']
female_performance = tennis[tennis['gender'] == 'Female']['perform']
ans2= ttest_ind(male_performance, female_performance)
pval2 = ans2.pvalue
alpha= 0.05
if pval2 < alpha:
    print("Reject H0,There is a difference in performance between males and females ")
else:
    print("Do not reject H0, There is no difference in performance between males and females")
#Do not reject H0, There is no difference in performance between males and females


#Create a boxplot to visualize performance across gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='perform', data=tennis, palette='coolwarm')
plt.title('Performance Distribution by Gender in Tennis')
plt.ylabel('Performance')
plt.xlabel('Gender')
plt.show()

#Create a bar plot to visualize the count of performance levels across genders
tennis['performance_category'] = pd.cut(tennis['perform'], 
                                    bins=[0, 50, 75, 100], 
                                    labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='performance_category', data=tennis, palette='coolwarm')
plt.title('Distribution of Performance Categories by Gender in Tennis')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.legend(title='Performance Category')
plt.show()

# Graph: Stacked Bar Plot
tennis3.plot(kind='bar', stacked=True, color=['#FFCC99', '#66CCFF', '#99FFCC'])
plt.title('Performance vs. Psychological Factors Categories in Tennis')
plt.xlabel('Performance Category')
plt.ylabel('Count')
plt.legend(title='Psychological Factors')
plt.savefig('tennis_stacked_bar_psych.png')
plt.close()
###########################################################################################################################################
###########################################################################################################################################
#Phase 3
tennis29=sport_perf[sport_perf.sport=="Tennis"]

# Define the dependent and independent variables
X = sm.add_constant(tennis29[['x13', 'x23', 'x22']]) # variables: fitness, psychological factors, hydration levels
y = tennis29['perform']

# Initial Multicollinearity Check: Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Initial Variance Inflation Factors:")
print(vif_data)

# Warn if high multicollinearity is detected
if (vif_data['VIF'][1:] > 10).any():  
    print("Warning: High multicollinearity detected in initial data (VIF > 10).")
else:
    print("No significant multicollinearity in initial data (all VIF < 10).")

# Initial linear regression model
try:
    initial_model = sm.OLS(y, X).fit()
    print("\nInitial Model Summary (before removing outliers/influential points):")
    print(initial_model.summary())
except Exception as e:
    print(f"Error fitting initial model: {e}")
    

# influential points using Cook's Distance
influence = initial_model.get_influence()
cooks_d = influence.cooks_distance[0]
n = len(tennis29)
cooks_threshold = 4 / n
influential_points = X.index[cooks_d > cooks_threshold]
print(f"\nInfluential points (Cook's Distance > {cooks_threshold:.4f}): {len(influential_points)} points")
print(f"Indices: {list(influential_points)}")

# Identify outliers using standardized residuals
standardized_residuals = influence.resid_studentized_internal
outlier_threshold = 3
outliers = X.index[abs(standardized_residuals) > outlier_threshold]
print(f"Outliers (Standardized Residuals > {outlier_threshold}): {len(outliers)} points")
print(f"Indices: {list(outliers)}")

# Combine influential points and outliers
points_to_remove = set(influential_points).union(set(outliers))
print(f"Total points to remove (influential or outliers): {len(points_to_remove)}")
print(f"Indices: {list(points_to_remove)}")

# Remove influential points and outliers
tennis_clean = tennis29.drop(index=points_to_remove).copy()
X_clean = tennis_clean[['x13', 'x23', 'x22']]
X_clean = sm.add_constant(X_clean)
y_clean = tennis_clean['perform']

# Fit model on cleaned data
try:
    cleaned_model = sm.OLS(y_clean, X_clean).fit()
    print("\nModel Summary (after removing outliers/influential points):")
    print(cleaned_model.summary())
except Exception as e:
    print(f"Error fitting cleaned model: {e}")
    exit()

# Check Model Assumptions for Cleaned Model
plt.figure(figsize=(8, 6))
sns.scatterplot(x=cleaned_model.fittedvalues, y=cleaned_model.resid, color='green')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals vs. Fitted Values in Tennis Model (Cleaned Data)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.savefig('tennis_residuals_fitted_clean.png')
#Homoscedasticity is observed, which is good.Residuals are fairly scattered and they have constant variance

plt.figure(figsize=(8, 6))
sm.qqplot(cleaned_model.resid, line='45', fit=True)
plt.title('Q-Q Plot of Residuals in Tennis Model (Cleaned Data)')
plt.savefig('tennis_qq_plot_clean.png')
#Based on the graphical information, Resids are normally distributed... No need to use jenkins box-cox transformation


# Model Refinement, Final Equation, and Prediction 

# Model Refinement: Check for multicollinearity in cleaned data
X_vif = tennis_clean[['x13', 'x23', 'x22']] 
vif_data_clean = pd.DataFrame()
vif_data_clean['Variable'] = X_vif.columns
vif_data_clean['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\nVariance Inflation Factors (Cleaned Data):")
print(vif_data_clean)

# remove x23 due to VIF > 10
if vif_data_clean.loc[vif_data_clean['Variable'] == 'x23', 'VIF'].values[0] > 10:
    print("\nRefining model by removing x23 due to high multicollinearity (VIF > 10).")
    X_refined = tennis_clean[['x13', 'x22']]  
    X_refined = sm.add_constant(X_refined)
    final_model = sm.OLS(y_clean, X_refined).fit()
    print("\nFinal Model Summary (after removing x23):")
    print(final_model.summary())
else:
    print("\nNo need to remove x23 (VIF <= 10). Using cleaned model as final model.")
    final_model = cleaned_model

# Final Model Equation
print("\nFinal Model for Tennis: Performance = {:.2f}".format(final_model.params['const']), end="")
var_labels = {'x13': 'Fitness', 'x23': 'Psychological', 'x22': 'Hydration'}
for var in [v for v in ['x13', 'x23', 'x22'] if v in final_model.params]:
    print(" + {:.2f}*{}".format(final_model.params[var], var_labels[var]), end="")
print()

# Predict performance using my new refined model
new_data_dict = {'const': 1, 'x13': 20, 'x23': 85, 'x22': 40}
new_data = pd.DataFrame({
    var: [new_data_dict[var]] for var in final_model.params.index
})
prediction = final_model.predict(new_data)
print(f"\nPredicted Performance for Tennis (x13=20, x23=85, x22=40): {prediction[0]:.2f}")



