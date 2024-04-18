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


# firefox 버전
# profile = webdriver.FirefoxProfile()
# profile.set_preference('general.useragent.override', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0')
# profile.set_preference("network.proxy.type", 1)
# profile.set_preference("network.proxy.socks", "127.0.0.1")
# profile.set_preference("network.proxy.socks_port", 9050)

#path = "G:/내 드라이브/Colab_code/Capstone/crawling/driver/geckodriver"
#driver = webdriver.Firefox(firefox_profile=profile, executable_path=path)
#driver = webdriver.Firefox(executable_path=path)

driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

# 정렬 방식 선택
# 1: 추천순
# 2: 최신순
# 기타: 정확도 순
def sort_kind(index):
    # 추천
    if index == 1:
        return 'vcount'
    # 최신순
    elif index == 2:
        return 'date'
    # 정확도
    else:
        return 'none'
driver.get('https://kin.naver.com/people/expert/index.naver?type=ANIMALDOCTOR')
# 크롤링 시작 일자
f = '2024.01.01'
# 크롤링 종료 일자
t = '2024.04.11'
period_txt = "&period=" + f + ".%7C" + t + "."

_sort_kind = sort_kind(2)
date = str(datetime.now()).replace('.', '_')
date = date.replace(' ', '_')
count = 0
# 전문가 리스트 URL 
f = open("G:/내 드라이브/Colab_code/Capstone/crawling/result/expert_list.txt", 'r')
expert_page = f.readlines()
f.close()
# 각 전문가 답변 URL 저장할 파일
f = open("G:/내 드라이브/Colab_code/Capstone/crawling/result/url_list.txt", 'w') # + keyword.replace(' ', '+') 
page_url = []

for expert in expert_page:
    page_index = 1
    while True:
        time.sleep(uniform(0.01, 1.0))
        #driver.get('https://kin.naver.com/userinfo/expert/answerList.naver?u=p0HH28AHsd3ylG26Io465ERBNm4nHsoAmxOB8l6hxog%3D' + "&sort=" + _sort_kind + "&section=kin" + "&page=" + str(page_index))
        #driver.get(expert + "&sort=" + _sort_kind + period_txt + "&section=kin" + "&page=" + str(page_index))  # 특정 기간만 크롤링 (수정 필요)
        driver.get(expert + "&sort=" + _sort_kind + "&section=kin" + "&page=" + str(page_index))  # 전부 크롤링
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        tags = soup.find_all('td', class_="title")
        if tags == []:
            break
        # <a href="/qna/detail.naver?d1id=8&amp;dirId=80511&amp;docId=443102661" rel="KIN">강아지 치은종</a>
        # find_all("div","corp_ara") : soup에서 div tag의 class가 corp_area인 태그를 가져온다.
        # <a></a>태그 : 하이퍼링크를 걸어주는 태그
        for tag in tags:
            href = tag.find('a')['href']
            url = 'https://kin.naver.com' + href
            page_url.append(url)
            f.write(url + "\n")
        
        if page_index == 500:
            break
        else:
            page_index += 1
f.close()
