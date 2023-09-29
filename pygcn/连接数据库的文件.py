import psycopg2

# 连接到PostgreSQL数据库
conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_user",
    password="your_password"
)

# 创建游标
cursor = conn.cursor()

# 执行SQL查询
cursor.execute("SELECT * FROM your_table")

# 获取查询结果
results = cursor.fetchall()

# 打印结果
for row in results:
    print(row)

# 关闭游标和连接
cursor.close()
conn.close()
