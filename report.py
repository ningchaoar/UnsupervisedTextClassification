from sklearn import metrics


def get_report(result_file: str, label_file: str):
    # 打印评测结果
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
