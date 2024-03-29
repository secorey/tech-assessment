from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import csv

# URL to scrape:
url = "https://lamadeleine.com/wp-json/wp/v2/restaurant-locations?per_page=150"

# Set options for web driver:
options = Options()
options.add_argument("--headless=new")
# Add path to chromedriver (change as necessary):
chrome_driver_path = "/Users/samcorey/anaconda3/lib/python3.11/" + \
    "site-packages/chromedriver_autoinstaller/120/chromedriver"
# Run driver and collect data:
driver = webdriver.Chrome(options=options, executable_path=chrome_driver_path)
driver.get(url)
json_data = driver.find_element_by_tag_name("pre").text
driver.quit()

# Read json data:
data_dict = json.loads(json_data)

# Write data to CSV:
with open ("location_data.csv", "w") as f:
    # Write column names:
    col_names = ["locationName", "streetAddress", "city", "state", "postalCode",
                "phoneNumber", "storeID"]
    writer = csv.DictWriter(f, fieldnames=col_names)
    writer.writeheader()
    # Iterate over locations and write relevant fields to CSV file:
    for location in data_dict:
        location_details = location["acf"]["locationHero"]
        row = {
            "storeID": location["id"],
            "locationName": location_details["storeName"],
            "streetAddress": location_details["addressLine1"],
            "city": location_details["city"],
            "state": location_details["state"],
            "postalCode": location_details["zip"],
            "phoneNumber": location_details["phone"],
        }
        writer.writerow(row)
