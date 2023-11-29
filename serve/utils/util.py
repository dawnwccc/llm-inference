from fastapi import Request


def get_real_ip(request: Request) -> str:
    # 获取 X-Forwarded-For 头
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    # 如果 X-Forwarded-For 头存在，取第一个 IP 地址作为客户端 IP
    if x_forwarded_for:
        real_ip = x_forwarded_for.split(",")[0].strip()
    else:
        # 如果 X-Forwarded-For 头不存在，则使用 request.client.host
        real_ip = request.client.host
    return real_ip


def check_crawler(request: Request) -> bool:
    # 1. 通过判断User-Agent来判断是否是爬虫
    crawlers = ["bot", "crawler"]
    user_agent = request.headers.get("User-Agent", "").lower()
    return any(crawler in user_agent for crawler in crawlers)
