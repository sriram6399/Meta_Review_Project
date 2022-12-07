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


def write_to_file(year,driver,des,url): 
    filename = "Data/ICLR/ICLR_"+des+"_"+year+".csv"
    driver.get(url)

    pag = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/nav/ul/li[14]/a')))
    desired_y = (pag.size['height'] / 2) + pag.location['y']
    current_y = (driver.execute_script('return window.innerHeight') / 2) + driver.execute_script('return window.pageYOffset')
    scroll_y_by = desired_y - current_y
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/nav/ul/li[14]/a'))).click()
    time.sleep(2)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/nav/ul/li[12]/a'))).click()
    time.sleep(2)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/ul/li')))
    list = driver.find_elements(By.XPATH,'/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/ul/li')
    rev= []
    for k in range(1,len(list)+1):
        w_rev =[]
        meta = ''
        try:
            driver.get(url)

            pag = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/nav/ul/li[14]/a')))
            desired_y = (pag.size['height'] / 2) + pag.location['y']
            current_y = (driver.execute_script('return window.innerHeight') / 2) + driver.execute_script('return window.pageYOffset')
            scroll_y_by = desired_y - current_y
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/nav/ul/li[14]/a'))).click()
            time.sleep(5)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/nav/ul/li[12]/a'))).click()
            time.sleep(3)
            
            element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/ul/li['+str(k)+']/h4/a[1]')))
            desired_y = (element.size['height'] / 2) + element.location['y']
            current_y = (driver.execute_script('return window.innerHeight') / 2) + driver.execute_script('return window.pageYOffset')
            scroll_y_by = desired_y - current_y
            driver.execute_script("window.scrollBy(0, arguments[0]);", scroll_y_by)
            
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/ul/li['+str(k)+']/h4/a[1]'))).click()
            
            reviews = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div')))
            #reviews = driver.find_elements(By.XPATH,'/html/body/div/div[3]/div/div/main/div/div[3]/div')
            print(len(reviews))
            for i in range(1,10+1):
                a = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div')))
            
                print(i)
                #a = driver.find_elements(By.XPATH,'/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div')
                for j in range(1,len(a)+1):
                    
                    print("hi"+str(j))
                    b = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div['+str(j)+']')))
                    desired_y = (b.size['height'] / 2) + b.location['y']
                    current_y = (driver.execute_script('return window.innerHeight') / 2) + driver.execute_script('return window.pageYOffset')
                    scroll_y_by = desired_y - current_y
                    
                    if(b.get_attribute("class")=="title_pdf_row clearfix"):
                        print("title1")
                        title = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div['+str(j)+']/h2/a'))).get_attribute("innerText")
                    
                    if(b.get_attribute("class")=="note_contents" and ("Comment" in (WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div['+str(j)+']/span[1]'))).text))):
                        
                        c = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div['+str(j)+']/span[2]'))).text
                        print(c)
                    if(b.get_attribute("class")=="note_contents" and ("Main Review" in (WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div['+str(j)+']/span[1]'))).text))):
                        
                        c = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/div/div/main/div/div[3]/div['+str(i)+']/div[1]/div['+str(j)+']/span[2]'))).text
                        print(c)

                if "Paper Decision" in title:  
                    print("test1")
                    meta = meta + c
                if "Official Review" in title:
                    print("test")
                    w_rev.append(c)

            
            
        except Exception as e:
            print(e)
            if len(w_rev) == 0 or len(meta)==0:
                print("failed")
                continue
            while len(w_rev)<3:
                w_rev.append('')
            print(meta)
            print(w_rev)
            
        print("cont")
        if len(w_rev) == 0 or len(meta)==0 or len(w_rev)<3:
            continue
        rev.append([w_rev[0],w_rev[1],w_rev[2],meta])

        time.sleep(0.5)
     
    driver.implicitly_wait(1000)
    print(meta)
    print(rev)
    with open(filename, 'a+', newline='',encoding="utf-8") as f:
        write = csv.writer(f,delimiter=',')
        write.writerows(rev)
    

driver = webdriver.Chrome(executable_path="Extra_Files\chromedriver.exe")
url = "https://openreview.net/group?id=ICLR.cc/2022/Conference#poster-submissions"
write_to_file("2022",driver,"poster18",url)