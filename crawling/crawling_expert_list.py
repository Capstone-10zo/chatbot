# required to install selenium, bs4 
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

import openpyxl
from openpyxl.styles import PatternFill, Color
from openpyxl import Workbook
from random import *
from webdriver_manager.firefox import GeckoDriverManager


# profile = webdriver.FirefoxProfile()
# profile.set_preference('general.useragent.override', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0')
# profile.set_preference("network.proxy.type", 1)
# profile.set_preference("network.proxy.socks", "127.0.0.1")
# profile.set_preference("network.proxy.socks_port", 9050)

#path = "G:/내 드라이브/Colab_code/Capstone/crawling/driver/geckodriver"
#driver = webdriver.Firefox(firefox_profile=profile, executable_path=path)
#driver = webdriver.Firefox(executable_path=path)

driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

# 전체 전문가 리스트 크롤링
page_index = 1
expert_page = []
f = open("G:/내 드라이브/Colab_code/Capstone/crawling/result/expert_list" + ".txt", 'w')
while True:
    time.sleep(uniform(0.01, 1.0))
    driver.get('https://kin.naver.com/people/expert/index.naver?type=ANIMALDOCTOR'+ "&page=" + str(page_index)) # 'https://kin.naver.com/search/list.nhn?query=' + get_keyword(keyword)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    pros = soup.find('ul',class_='pro_list')
    tags = pros.find_all('li')
    for tag in tags:
        expert_url = 'https://kin.naver.com/' + tag.find_all('a')[-1]['href']
        expert_page.append(expert_url)
        f.write(expert_url + "\n")
    if page_index == 5:
            break
    else:
        page_index += 1
f.close()
    