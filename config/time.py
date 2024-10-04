import random
from datetime import datetime, timedelta

# 定义时间范围
start_date = datetime(2021, 11, 22)
end_date = datetime(2023, 4, 10)

# 计算总时间
total_hours = 727

# 生成时间段
time_slots = []
current_date = start_date
while sum(len(slot) for slot in time_slots) < total_hours:
    # 生成随机时间段
    start_hour = random.randint(18, 23)
    end_hour = random.randint((start_hour + 1) - 23, 8)
    start_minute = random.randint(0, 59)
    end_minute = random.randint(0, 59)

    # 确保时间段在1-5小时内
    duration = (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute)
    if duration < 60 or duration > 300:
        continue

    # 生成时间段字符串
    time_slot = f"{current_date.strftime('%Y/%m/%d/%H:%M')}-{current_date.strftime('%Y/%m/%d/%H:%M')}"
    time_slots.append(time_slot)

    # 更新下一个时间段的日期
    current_date += timedelta(hours=random.randint(1, 5))

    # 如果超过结束日期,则退出循环
    if current_date > end_date:
        break

# 输出结果
for slot in time_slots:
    print(slot)