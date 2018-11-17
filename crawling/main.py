# coding=UTF-8

import requests
from lxml import html
# from urllib.request import urlopen#用于获取网页
from bs4 import BeautifulSoup#用于解析网页
import json
from lxml import etree
import urllib2

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def get_url(url):
    req = urllib2.Request(url)
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    response = urllib2.urlopen(req)
    #tml = res.read().decode('utf-8')
    html = response.read()
    selector = etree.HTML(html)
    print html.decode('utf-8')
    data = selector.xpath("a")
    news = []
    for i in data:
        news.append(i["href"])

    #//a[@class='single-story-module__headline-link']|//section/section[@id='_up_with_heds_1']/div/article/h3/a



    #
    # # 请求URL，获取其text文本
    # wbdata = requests.get(url).text
    # # 对获取到的文本进行解析
    # soup = BeautifulSoup(wbdata, 'lxml')
    # # 从解析文件中通过select选择器定位指定的元素，返回一个列表
    # news_links = soup.select("h3")
    # #,h3.story-package-module__story__headline > a
    # #.single-story-module__headline-link
    # news = []
    #
    # # 对返回的列表进行遍历
    # for n in news_links:
    #     # 提取出标题和链接信息
    #     link = n.get("href")
    #     news.append(link)
    print news
    return news


def get_content(url):
    req = urllib2.Request(url)
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    response = urllib2.urlopen(req)
    html = response.read()
    selector = etree.HTML(html)


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
    title = soup.select("//h1[@class='lede-text-v2__hed']")
    content_json["title"] = title.get_text()

    # author
    author = soup.select("//address[@class='lede-text-v2__byline']/div[@class='author-v2']/a[@class='author-v2__byline']")
    content_json["author"] = author.get_text()

    # time
    time = soup.select("//div[@class='lede-text-v2__times']/time[@class='article-timestamp']")
    content_json["time"] = time.get_text()

    # abstract
    abstract = soup.select("//div[@class='abstract-v2__item-text']")
    content_json["abstract"] = abstract.get_text()

    # content
    content = soup.select("//div[@class='body-copy-v2 fence-body']/p|//img[@class='lazy-img__image loaded']")
    para = []
    for n in content:
        tmp = n.get_text()
        if tmp: # p
            data = selector.xpath("//div[@class='body-copy-v2 fence-body']/p")
            p = data[0].xpath('string(.)').extract()[0]
            p = p[1:len(p)-1] # 去除头尾"
            flag = 0
            while flag == 0:
                cur = 0
                for i in range(len(p)-1):
                    if p[i] == '\"' & p[i+1] == '\"':
                        cur = i
                        break
                if cur == 0:
                    flag = 1
                else:
                    p = p[0:i] + p[i+2:len(p)]
            para.append(p)
        else: # img
            img = n.get("src")
            para.append(["img",img])
    content_json["content"] = json.dumps(para)

    return content_json


main_url = 'https://www.bloomberg.com/markets'
news_url = get_url(main_url)

# for j in news_url:
#     content_crawl = get_content(j)
#     # save jason
#     file1 = open('./news/'+j["title'"]+'.json', 'w', encoding='utf-8')
#     json.dump(j, file1, ensure_ascii=False)
#     file1.close()

