import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pandas as pd
import os
import time
from tqdm import tqdm  # 导入 tqdm

# URL 模板，最后的数字可以变化
url_template = "https://www.rtl.lu/mobiliteit/news/a/{0}.html"


# 检查是否为 404 页面
def is_page_not_found(soup):
    return "404" in soup.text


def extract_date(soup):
    metainfo = soup.find("div", class_="article-heading__metainfo")
    if metainfo:
        return metainfo.get_text(strip=True)
    return "日期信息未找到"


def extract_date_details(date_string):
    # 匹配日期的正则表达式
    match = re.search(r"Update:\s*(\d{2})\.(\d{2})\.(\d{4})", date_string)
    if match:
        day, month, year = map(int, match.groups())
        try:
            # 创建标准 datetime 对象并返回为字符串
            date_obj = datetime(year, month, day)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            return "日期格式无效"
    return "日期信息未找到"


def scrape_article_content(article_number):
    url = url_template.format(article_number)
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # 检测 404 页面
        if is_page_not_found(soup):
            print(f"文章 {article_number} 不存在（404 页面）。")
            return {"url": url, "error": "404 Not Found"}

        # 提取日期信息
        date_info = extract_date(soup)
        date_details = extract_date_details(date_info) if date_info else None

        # 找到文章主体内容
        article_body = soup.find("div", class_="article__body")
        content = []
        if article_body:
            paragraphs = article_body.find_all("p")
            content_list = [paragraph.get_text().strip() for paragraph in paragraphs]
            content = "\n".join(
                paragraph.get_text().strip() for paragraph in paragraphs
            )

        # 返回结果
        return {
            "url": url,
            "date_info": date_details,
            "content": content,
            "content_list": content_list,
            "error": "NONE",
        }
    else:
        # print(f"无法获取文章 {article_number}，状态码: {response.status_code}")
        return {
            "url": url,
            "date_info": "",
            "content": "",
            "content_list": "",
            "error": f"HTTP {response.status_code}",
        }


# 2160001 - > 23.01.2024
# 2260902 - > 20.12.2024
# 示例：提取文章内容
scrape_article_content(2260902)
# scrape_article_content(2260802)
# 如果需要处理多个文章，可以用循环
csv_file = "articles.csv"

# 如果文件不存在，先创建并写入表头
if not os.path.exists(csv_file):
    columns = ["url", "date_info", "content", "content_list", "error"]
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

# 遍历所有文章编号
for article_number in tqdm(range(2160001, 2260903), desc="Processing articles"):
    response_dict = scrape_article_content(article_number)

    # 将字典转换为 DataFrame
    df = pd.DataFrame([response_dict])

    # 逐条追加到 CSV 文件中，不写入索引和表头
    df.to_csv(csv_file, mode="a", header=False, index=False)

    # 每次请求后暂停 5 秒
    time.sleep(1)
    # print(f"Saved article {article_number} to CSV.")
print("Data has been saved to 'articles.csv'.")
