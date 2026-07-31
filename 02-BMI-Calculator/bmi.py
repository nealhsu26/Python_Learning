height = float(input('请输入身高(m): '))
weight = float(input('请输入体重(kg): '))
bmi = weight / (height ** 2)
if bmi < 18.5:
    print("体重过轻")
elif bmi < 24:
    print("体重正常")
elif bmi < 28:
    print("超重")
else:
    print("肥胖")