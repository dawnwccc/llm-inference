import asyncio
import time


# 定义异步函数
async def async_function():
    print("Start async function")
    # 模拟异步操作，这里使用 asyncio.sleep 来模拟 I/O 操作
    time.sleep(1)
    print("Async function completed")


# 异步调用
async def main():
    print("Before calling async function")

    # 使用 await 调用异步函数
    await async_function()

    print("After calling async function")


# 运行异步程序
if __name__ == "__main__":
    # 创建一个事件循环
    loop = asyncio.get_event_loop()

    # 运行主函数，直到完成
    loop.run_until_complete(main())
