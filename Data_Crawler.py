from re import A
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import csv
import time
from selenium.webdriver.firefox.options import Options

options = Options()
options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'


def write_to_file(year):
    filename = "Data/NIPS/NIPS_"+year+".csv"
    rev= []
    driver = webdriver.Firefox(executable_path=r'Extra_Files\geckodriver.exe', options=options)
    driver.get('https://papers.nips.cc/paper/'+year)
    items = driver.find_elements(By.XPATH, '/html/body/div[2]/div/ul/li')
        
    for i in range(0,len(items)): 
        try:
            driver.get('https://papers.nips.cc/paper/'+year)
            target = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div/ul/li['+str(i+1)+']/a')))
            target.location_once_scrolled_into_view
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div/ul/li['+str(i+1)+']/a'))).click()
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div/div/a[4]'))).click()
            rev1 = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]'))).text
            rev2= WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[3]'))).text
            rev3 = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[4]'))).text
            if rev1 != '':
                rev.append([rev1,rev2,rev3])
            time.sleep(0.5)
        except Exception as  e:
            print(e)
            if rev1 != '':
                rev.append([rev1,rev2,rev3])
            time.sleep(0.5)
        
    driver.implicitly_wait(1000)

    with open(filename, 'w', newline='',encoding="utf-8") as f:
        write = csv.writer(f,delimiter=',')
        write.writerows(rev)

write_to_file("2017")