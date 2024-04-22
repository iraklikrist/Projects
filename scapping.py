from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import re
import matplotlib.pyplot as plt


driver = webdriver.Chrome()

product_links = []
for i in range(1, 10):
    url = f'https://www.mymarket.ge/en/search/78/technic/audio-video-and-photo-equipment/tv/?CatID=78&Keyword=tv&Limit=10&Limit=10&OfferPrice=0&Page={i}&SetTypeID=2'
    driver.get(url)
    time.sleep(30)
    elements = driver.find_elements(By.CSS_SELECTOR, '.h-100[href]')
    for element in elements:
        product_links.append(element.get_attribute('href'))

product_list = []
for link in product_links:
    url = link
    if url != "https://www.mymarket.ge/en":
        # Initialize variables outside of the loop
        product_name = product_price = product_display = product_adapt = product_resolution = product_manufacturer = product_type = ""
        product_brand = product_3D = product_resolution = product_size = product_smart = product_type = product_technology = product_adapt = ""

        driver.get(url)
        print(url)
        time.sleep(8)
        product_main = driver.find_elements(By.ID, 'width_id')
        product_spec = driver.find_elements(By.CLASS_NAME, 'spec-list')
        for element in product_main:
            product_name = element.find_element(By.CLASS_NAME, 'font-bold').text
            product_price = element.find_element(By.CLASS_NAME, 'font-size-24').text
            product_link = link
            product_price = product_price[:-1]
            print(product_name)
            print(product_price)

        for element in product_spec:
            specifiactions = element.find_elements(By.CLASS_NAME, 'd-flex')
            for e in specifiactions:
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Brand":
                    product_brand = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_brand)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "3D":
                    product_3D = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_3D)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Resolution":
                    product_resolution = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_resolution)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Screen Size":
                    product_size_text = e.find_element(By.CSS_SELECTOR, 'p').text
                    # Extract numeric part from 'Size' using regular expression
                    size_numeric = re.search(r'\d+', product_size_text).group()
                    product_size = int(size_numeric) if size_numeric else None
                    print(product_size)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Smart TV":
                    product_smart = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_smart)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Type":
                    product_type = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_type)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Display Technology":
                    product_technology = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_technology)
                if e.find_element(By.CSS_SELECTOR, 'span').text == "Adapted for PSN":
                    product_adapt = e.find_element(By.CSS_SELECTOR, 'p').text
                    print(product_adapt)

            product_list.append(
                [product_link, product_name, product_price, product_brand, product_3D,
                 product_resolution, product_size, product_smart, product_type, product_technology, product_adapt])


driver.quit()

# Corrected column names in the DataFrame
products = pd.DataFrame(product_list, columns=["Link","Name", "Price", "Brand", "3D", "Resolution", "Size", "Smart", "Type", "Technology", "Adapted for PSN"])
products.to_excel('monitors2.xlsx', index=False)







file_path = 'monitors2.xlsx'  # Replace with the actual path to your Excel file
df = pd.read_excel(file_path)

# Extract the 'Price' column
prices = df['Price']

# Calculate the standard deviation
std_deviation_price = prices.std()

# Print the result
print(f"Standard Deviation for 'Price': {std_deviation_price}")


median_price = prices.median()

# Print the result
print(f"50th Percentile (Median) for 'Price': {median_price}")





# Corrected column names in the DataFrame
products = df
# Drop rows with NaN in 'Size' column
products = products.dropna(subset=['Size'])

# Convert 'Size' to int
products['Size'] = products['Size'].astype(int)

# Save the DataFrame to an Excel file
products.to_excel('monitors2.xlsx', index=False)

# Regression Model
# Assuming 'Size' is the independent variable and 'Price' is the dependent variable
X = products[['Size']]
y = products['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the Root Mean Squared Error (RMSE)
print(f"Root Mean Squared Error: {rmse}")

# Print the coefficients and intercept of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

r2 = r2_score(y_test, y_pred)

# Print the R-squared score
print(f"R-squared Score: {r2}")

if r2 < 0.5:
    print("R-squared Score: Low")
elif 0.5 <= r2 < 0.75:
    print("R-squared Score: Medium")
elif 0.75 <= r2 < 0.9:
    print("R-squared Score: Decent")
else:
    print("R-squared Score: Good")
X = products[['Size']]
y = products['Price']

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the original data
y_pred = model.predict(X)

# Plot the original data points
plt.scatter(X, y, label='Actual Data')

# Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=3, label='Linear Regression Line')

# Labeling the axes and adding a title
plt.title('Linear Regression: Size vs Price')
plt.xlabel('Size')
plt.ylabel('Price')

# Display the legend
plt.legend()

# Show the plot
plt.show()





def get_top_tvs_within_budget(products_df, budget):
    # Convert 'Price' and 'Resolution' columns to numeric values, handling errors by setting them to NaN
    products_df['Price'] = pd.to_numeric(products_df['Price'], errors='coerce')
    products_df['Size'] = pd.to_numeric(products_df['Size'], errors='coerce')

    # Filter TVs based on budget
    budget_df = products_df[products_df['Price'] <= budget]

    # Return top 5 TVs with highest resolution
    top_tvs = budget_df.nlargest(5, 'Size')
    return top_tvs



user_budget = float(input("Enter your budget for the TV: "))

# Get top 5 TVs with the highest resolution within the user's budget
top_tvs = get_top_tvs_within_budget(products, user_budget)

# Display the top 5 TVs
print("\nTop 5 TVs within your budget with highest resolution:")
for index, row in top_tvs.iterrows():
    print(f"Link: {row['Link']}")
    print(f"\nTV {index + 1} Details:")
    print(f"Name: {row['Name']}")
    print(f"Price: {row['Price']}")
    print(f"Resolution: {row['Resolution']}")
    print(f"Brand: {row['Brand']}")
    print(f"Size: {row['Size']}")
