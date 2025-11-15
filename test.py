from openai import OpenAI

if __name__ == '__main__':
    client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.zhizengzeng.com/v1',
        api_key='sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6',
    )

    resp = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello!",
            }
        ],
        model="gpt-4o-mini", # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动
    )

    print(resp.choices[0].message.content)
