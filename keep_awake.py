from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup a Headless Chrome browser
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)

try:
    driver.get("https://nbatop100modeling-tqyjgjsniaxnc5vtwkaulb.streamlit.app/")
    print("Visited the app.")
    
    # Wait up to 10 seconds to see if the wake-up button appears
    wait = WebDriverWait(driver, 10)
    button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Yes, get this app back up')]")))
    button.click()
    print("App was asleep. Clicked the wake-up button!")
except Exception as e:
    print("App is already awake or button not found. All good!")
finally:
    driver.quit()