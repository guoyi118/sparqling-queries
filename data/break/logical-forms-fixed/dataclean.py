import jsonlines
import pandas as pd

data = []
question = []
qid = []

def clean(code):
    cleaned_code = []
    for op in code:
        if op[0] == 'select':
            args = op[1][0]["arg"]['type'] + ':' + op[1][0]["arg"]['keys'][0]
            # [arg["arg"]['type'] for arg in op[1]]
            cleaned_code.append(op[0] + ' ' + args)

        elif op[0] == 'comparative':
            ref1 = op[1][0]['arg'][0]
            ref2 = op[1][1]['arg'][0]
            if ref1 == ref2 :
                try:
                    operation = op[1][2]['arg']['keys'][0] + op[1][2]['arg']['keys'][1]
                    attribute = op[1][2]['arg']['keys'][2]['type'] + ':' + op[1][2]['arg']['keys'][2]['keys'][0] + '/' + op[1][2]['arg']['keys'][2]['keys'][1]
                    cleaned_code.append(op[0] + ' ' + ref1 + ' '+ ref2 + ' '+ operation + ' ' + attribute )
                except:
                    if op[1][2]['arg'] is None:
                        attribute = "none"
                    else:
                        attribute = op[1][2]['arg']['type'] + ':' + '/'.join(op[1][2]['arg']['keys'])
                    cleaned_code.append(op[0] + ' ' + ref1 + ' '+ ref2 + ' ' + attribute )
            else:
                if op[1][2]['arg'] is None:
                    operation = 'none'
                else:
                    try:
                        operation = op[1][2]['arg']['keys'][0] + op[1][2]['arg']['keys'][1]
                    except:
                        operation = op[1][2]['arg']['type'] + ':' + op[1][2]['arg']['keys'][0]
                cleaned_code.append(op[0] + ' ' + ref1 + ' '+ ref2 + ' '+ operation)
        elif op[0] == 'aggregate':
            try:
                operator = op[1][0]['arg']
                ref = op[1][1]['arg'][0]
                cleaned_code.append(op[0] + ' ' + operator + ' ' + ref )
            except: 
                attribute = op[1][0]['arg']['type'] + ':' + '/'.join(op[1][0]['arg']['keys'])
                ref = op[1][1]['arg'][0]
                cleaned_code.append(op[0] + ' ' + attribute + ' ' + ref )
        elif op[0] == 'project':
            if op[-1]:
                continue
            if op[1][0]['arg'] is None:
                attribute = "none"
            else:
                attribute = op[1][0]['arg']['type'] + ':' + '/'.join(op[1][0]['arg']['keys'])
            ref = op[1][1]['arg'][0]
            cleaned_code.append(op[0] + ' ' + attribute + ' ' + ref )
        elif op[0] == 'union':
            refs = [ref['arg'][0] for ref in op[1]]
            cleaned_code.append(op[0] + ' ' + ' '.join(refs))
        elif op[0] == 'group':
            operator = op[1][0]['arg']
            ref1 = op[1][1]['arg'][0]
            ref2 = op[1][2]['arg'][0]
            cleaned_code.append(op[0] + ' ' + ref1 + ' ' + ref2)
        elif op[0] == 'superlative':
            operator = op[1][0]['arg']
            ref1 = op[1][1]['arg'][0]
            ref2 = op[1][2]['arg'][0]
            cleaned_code.append(op[0] + ' ' + ref1 + ' ' + ref2)
        elif op[0] == 'intersection':
            refs = [ref['arg'][0] for ref in op[1]]
            cleaned_code.append(op[0] + ' ' + ' '.join(refs))
        elif op[0] == 'sort':
            try:
                ref1 = op[1][0]['arg'][0]
                ref2 = op[1][1]['arg'][0]
                sort = op[1][2]['arg']['keys'][0]
                cleaned_code.append(op[0] + ' ' + ref1 + ' ' + ref2 + ' ' + sort)
            except:
                ref1 = op[1][0]['arg'][0]
                ref2 = op[1][1]['arg'][0]
                # sort = op[1][2]['arg']['keys'][0]
                cleaned_code.append(op[0] + ' ' + ref1 + ' ' + ref2)
        elif op[0] == 'discard':
            ref1 = op[1][0]['arg'][0]
            ref2 = op[1][1]['arg'][0]
            cleaned_code.append(op[0] + ' ' + ref1 + ' ' + ref2)

        else:
            print(op)
            print(sadsd)
            # cleaned_code.append(op)
    
    return ", ".join(cleaned_code)

            



# with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/data/break/logical-forms-fixed/train.jsonl") as f:
#     for d in f:
#         qid.append(d["subset_idx"])
#         question.append(d["text"])
#         data.append(clean(d["qdmr_code"][0]))
#         # print(d["qdmr_code"])

# # print(pd.DataFrame(data))
# train_data = [pd.DataFrame(qid), pd.DataFrame(question), pd.DataFrame(data)]
# train_data_header =  ["id","source_text", "target_text"]
# train_data_df = pd.concat(train_data, axis=1, keys=train_data_header)
# train_data_df.to_csv('train_data_df.csv', index=False)


with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/data/break/logical-forms-fixed/val.jsonl") as f:
    for d in f:
        qid.append(d["subset_idx"])
        question.append(d["text"])
        data.append(clean(d["qdmr_code"][0]))
        # print(d["qdmr_code"])

# print(pd.DataFrame(data))
val_data = [pd.DataFrame(qid), pd.DataFrame(question), pd.DataFrame(data)]
val_data_header =  ["id","source_text", "target_text"]
val_data_df = pd.concat(val_data, axis=1, keys=val_data_header)
val_data_df.to_csv('val_data_df_spider.csv', index=False)


# for i in range(len(data)):
#     try:
#         clean(data[i])
#     except:
#         print(i)
#         print(sddssdds)


# print(question[1214])
# # print(data[13])
# print(clean(data[1214]))