# coding=UTF-8

import requests
from lxml import html
# from urllib.request import urlopen#用于获取网页
from bs4 import BeautifulSoup#用于解析网页
import json
from lxml import etree
import urllib2
import time,random#导入包
import io
import datetime
import MySQLdb

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def get_url(url):
    # req = urllib2.Request(url)
    # req.add_header('User-Agent',
    #                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    # response = urllib2.urlopen(req)
    # #tml = res.read().decode('utf-8')
    # html = response.read()
    # selector = etree.HTML(html)
    # # print html.decode('utf-8')
    # data = selector.xpath("//div[@class='story-content']/a/h3/text()")
    # news = []
    # for i in data:
    #     news.append(i)

    #//a[@class='single-story-module__headline-link']|//section/section[@id='_up_with_heds_1']/div/article/h3/a




    # 请求URL，获取其text文本
    wbdata = requests.get(url).text
    # 对获取到的文本进行解析
    soup = BeautifulSoup(wbdata, 'lxml')
    # 从解析文件中通过select选择器定位指定的元素，返回一个列表
    news_links = soup.select(".story-content a")
    #,h3.story-package-module__story__headline > a
    #.single-story-module__headline-link
    news = []

    # 对返回的列表进行遍历
    for n in news_links:
        # 提取出标题和链接信息
        link = n.get("href")
        news.append('https://www.reuters.com'+link)
    # print news
    return news


def get_content(url):
    # req = urllib2.Request(url)
    # req.add_header('User-Agent',
    #                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    # response = urllib2.urlopen(req)
    # html = response.read()
    # selector = etree.HTML(html)


    # # title
    # title = tree.xpath('//h1[@class='lede-text-v2__hed']/text()')
    #
    # # author
    # author = tree.xpath('//address[@class='lede-text-v2__byline']/div[@class='author-v2']/a[@class='author-v2__byline']')
    #
    # # time
    # time = tree.xpath('//div[@class='lede-text-v2__times']/time[@class='article-timestamp']')
    #
    # # abstract
    # abstract = tree.xpath('//div[@class='abstract-v2__item-text']')
    #
    # # abstract
    # abstract = tree.xpath('//div[@class='abstract-v2__item-text']')


    # 请求URL，获取其text文本
    wbdata = requests.get(url).text

    # 对获取到的文本进行解析
    soup = BeautifulSoup(wbdata, 'lxml')

    content_json = {}

    # title
    title = soup.select(".ArticleHeader_headline")
    content_json["title"] = title[0].text

    # author
    author = soup.select(".BylineBar_byline")
    if len(author):
        content_json["author"] = author[0].text
    else:
        content_json["author"] = "Anonymous"

    # time
    time = soup.select(".ArticleHeader_date")
    if len(time):
        content_json["time"] = time[0].text.split(' /')[0]
    else:
        content_json["time"] = datetime.datetime.now().strftime('%m %d, %Y')

    # # abstract
    # abstract = soup.select("//div[@class='abstract-v2__item-text']")
    # content_json["abstract"] = abstract.get_text()

    # content
    content = soup.select(".StandardArticleBody_body p")
    para = []
    for n in content:
        para.append(n.get_text())

    content_json["content"] = json.dumps(para)

    return content_json


main_url = 'https://www.reuters.com/finance/markets'
news_url = get_url(main_url)

for j in news_url:
    content_crawl = get_content(j)
    # save jason
    with io.open('./news/'+ content_crawl["title"].replace("/", " ")+'.json', 'w', encoding="utf-8") as outfile:
        outfile.write(unicode(json.dumps(content_crawl, ensure_ascii=False)))

    # file1 = io.open('./news/'+ content_crawl["title"]+'.json', 'w', encoding='utf-8')
    # unicode(json.dump(j, file1, ensure_ascii=False))
    # file1.close()


# tmp = get_content("https://www.reuters.com/article/italy-budget-edp/update-2-eu-executive-to-launch-action-over-italy-budget-on-wednesday-sources-idUSL8N1XR3U9")
# # file1 = io.open('test'+'.json', 'w', encoding='utf-8')
# print json.dumps(tmp)
