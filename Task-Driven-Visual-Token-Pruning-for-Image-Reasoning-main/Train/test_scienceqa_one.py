from datasets import load_dataset

# 加载数据集
ds = load_dataset("/root/autodl-tmp/datasets/scienceqa")

# 遍历测试集并输出样本信息
for x in ds["test"]:
    if x["image"] is not None:
        print(x["question"])
        print(x["choices"])
        print(x["answer"])

        # 保存图片至临时目录
        x["image"].save("/tmp/scienceqa_test.png")
        
        # 处理完第一个有效样本后跳出循环
        break