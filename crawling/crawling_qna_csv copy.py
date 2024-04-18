from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains as ac

import requests
from bs4 import BeautifulSoup

import pandas as pd
import csv
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from webdriver_manager.firefox import GeckoDriverManager


# Driver Setting - firefox
'''
profile = webdriver.FirefoxProfile()

profile.set_preference('general.useragent.override', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0')
profile.set_preference("network.proxy.type", 1)
profile.set_preference("network.proxy.socks", "127.0.0.1")
profile.set_preference("network.proxy.socks_port", 9050)

path = "./driver/geckodriver"
driver = webdriver.Firefox(firefox_profile=profile, executable_path=path)
'''

# Driver Setting - chrome
'''
from webdriver_manager.chrome import ChromeDriverManager
chrome_options = Options()
#chrome_options.add_argument('headless')  # 브라우저 창을 띄우지 않고 실행할 경우
# 불필요한 에러 메시지 없애기
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
service = Service(executable_path=ChromeDriverManager().install())
browser = webdriver.Chrome(service=service, options=chrome_options, executable_path="G:/내 드라이브/Colab_code/Capstone/crawling/driver/chromedriver")
driver = browser
'''

import time
from datetime import datetime, timedelta
from random import *

driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
driver.page_source.encode('utf-8')


## read url list in file
f = open("G:/내 드라이브/Colab_code/Capstone/crawling/result/url_list.txt", 'r')
page_url = f.readlines()
f.close()
## crawling each url page

## crawling each url page
try:
    f = open("./QnA_data.csv", "a", encoding='utf-8')
    fieldnames = ['idx','title', 'question', 'answer', 'url']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    start_idx = 0
    for url in page_url[0:-1]:
        
        driver.get(url)
        # Question list
        # alert창 처리
        try:
            result = driver.switch_to.alert
            print(f"{start_idx}: result", result)
            result.accept()
            # csv에 저장
            import csv
            data = {'idx':start_idx,'title':'@Exception_alert@', 'question':"", 'answer':"", 'url':url}
            writer.writerow(data) 
            start_idx += 1
            continue
        except Exception as e:
            # print('alert창 예외가 발생했습니다.', e)  
            pass
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'question-content')))
            pros = driver.find_element(By.CLASS_NAME,'question-content')
            title = pros.find_element(By.CLASS_NAME,'title').text
            print(f"{start_idx}: title", title)
        except Exception as e:
            title = "@Exception_title@"
            print(f"{start_idx}: title", title)
            print('title: 예외가 발생했습니다.', e)  
            
        try :
            pros = driver.find_element(By.CLASS_NAME,'question-content')
            tags = pros.find_element(By.CLASS_NAME, 'c-heading__content')
            question_txt = tags.text
        except NoSuchElementException:
            question_txt = ""
        except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
            print('question_txt: 예외가 발생했습니다.', e)
            
        # answer_list = driver.find_elements(By.CLASS_NAME, "se-main-container")
        answer_list = driver.find_elements(By.CLASS_NAME, "answer-content__item")
        t = ""
        new_answer_list = []
        try:
            for n, answer in enumerate(answer_list):
                #texts = answer.find_elements_by_tag_name('span')
            
                profile = answer.find_element(By.CLASS_NAME, 'profile_card').text
                if profile.find('수의사') == -1:
                    continue
                new_answer_list.append(answer.find_element(By.CLASS_NAME, '_endContentsText').text)
            
            if len(new_answer_list) >= 2:
                print('수의사 답변이 2개 이상입니다.')
                t = "@Excepiton 수의사 답변이 2개 이상입니다."
            else:
                t = new_answer_list[0]
                
        except Exception as e:
            t = "@Excepiton"
            print('answer: 예외가 발생했습니다.', e)   
            # print('answer',t)
            
        
        # csv에 저장
        import csv

        data = {'idx':start_idx,'title':title, 'question':question_txt, 'answer':t, 'url':url}

        writer.writerow(data) 
        
        start_idx += 1
        
except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
    print('###### 예외가 발생했습니다', e)
finally:
    f.close()