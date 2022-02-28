import jsonlines
from ast import literal_eval
import pandas as pd 

def sentencenormalization(input):
    pased_input = literal_eval(input)
    output = [d.replace("["," ").replace("]","").replace("'","").replace(",","")  for d in pased_input]
    return ', '.join(output)


dev_df = pd.read_csv('/root/sparqling-queries/data/break/logical-forms-fixed/dev.csv')
dev_df['program'] = dev_df['program'].apply(sentencenormalization)

dev_df.to_csv('/root/sparqling-queries/data/break/logical-forms-fixed/dev_normalized.csv')


# with jsonlines.open("/root/sparqling-queries/data/break/logical-forms-fixed/train_alter_v2.jsonl") as f:
#     for d in f:
#         print(sentencenormalization(d["alternatives"]))
        # print(
        #     literal_eval(d["prediction"])[1].replace("["," ").replace("]"," ").replace("'","").replace(",","") + ","
        #     )

        # data.append(
        #     {
        #         k: d[k]
        #         for k in (
        #             "input",
        #             # 问题
        #             "prediction",
        #             # 模型输出（错误）
        #             "alternatives",
        #             #模型可能输出的其他答案
        #         )
        #     }
        # )

