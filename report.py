from sklearn import metrics


def get_report(result_file: str, label_file: str):
    """
    打印评测结果
    如格式不同, 调整代码并将标注和预测结果传入metrics.classification_report即可
    :param result_file: 训练后得到的文本分类结果
    :param label_file: 模型保存路径下的labels.txt路径
    :return:
    """
    with open(label_file, "r", encoding="utf-8") as fr:
        labels = [line.strip() for line in fr]
    with open(result_file, "r", encoding="utf-8") as fr:
        fr = [line.strip().rsplit("\t", 1) for line in fr]
    y_predict = []
    y_label = []
    for line in fr:
        raw_data, y = line
        label = raw_data.split("_!_")[2]
        y_predict.append(labels.index(y))
        y_label.append(labels.index(label))
    print(metrics.classification_report(y_label, y_predict, target_names=labels))


if __name__ == "__main__":
    get_report(result_file="resources/cropus/toutiao_cat_data_result.txt",
               label_file="resources/model/toutiao_news_model/labels.txt")
