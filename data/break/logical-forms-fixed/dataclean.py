import jsonlines
import pandas as pd
import inflect

def qdmr_clean(code):
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
                    operation = operation.replace('!=', 'notequal/').replace('<=', 'lessequal/').replace('>=', 'greatequal/').replace('=', 'equal/').replace('>', 'great/').replace('<', 'less/')
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
                operation = operation.replace('!=', 'notequal/').replace('<=', 'lessequal/').replace('>=', 'greatequal/').replace('=', 'equal/').replace('>', 'great/').replace('<', 'less/')
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
            cleaned_code.append(op[0] + ' ' + operator + ' ' +  ref1 + ' ' + ref2)
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

def input_combine(text, column_data):
    column_name = []
    for dt in column_data:
        table_name = list(dt.keys())
        for table in table_name:
            column_name = column_name + list(dt[table].keys())
            
        
    return text + ' <tbl> ' + ' '.join(table_name) + ' </tbl>' + ' <col> ' + ' '.join(column_name) + ' </col>'


def input_abstract(text, column_data, qdmr):
    data_attributes = {}
    cat_idx = 0
    num_idx = 0
    temp_idx = 0
    table_name = []
    column_name = []
    engine = inflect.engine()


    for dt in column_data:
        tables = list(dt.keys())
        for i in range(len(tables)):
            data_attributes[tables[i]] = "<tbl%s>"%(i)
            table_name.append("<tbl%s>"%(i))
        
        for table in tables:
            column = list(dt[table].keys())
            for j in column:
                if dt[table][j] == 'number':
                    replace = "<NUM%s>"%(num_idx)
                    data_attributes[j] = replace
                    num_idx += 1
                elif dt[table][j] == 'time':
                    replace = "<TEP%s>"%(temp_idx)
                    data_attributes[j] = replace
                    temp_idx += 1
                else:
                    replace = "<CAT%s>"%(cat_idx)
                    data_attributes[j] = replace
                    cat_idx += 1   

                column_name.append(replace)    
        
        text_token = text.split()

        for i in range(len(text_token)):
            for j in list(data_attributes.keys()):
                if text_token[i].lower() == j.lower() or engine.plural(text_token[i]).lower() == j.lower() or engine.plural(j).lower() == text_token[i].lower():
                    text = text.replace(text_token[i], data_attributes[j])



        # print(text + ' <tbl> ' + ' '.join(table_name) + ' </tbl>' + ' <col> ' + ' '.join(column_name) + ' </col>')

        qdmr_token = qdmr.split()

        for i in range(len(qdmr_token)):
            for j in list(data_attributes.keys()):
                if qdmr_token[i].lower() in j.lower() or j.lower() in qdmr_token[i]:
                    qdmr = qdmr.replace(qdmr_token[i], data_attributes[j] )
        


    return text + ' <tbl> ' + ' '.join(table_name) + ' </tbl>' + ' <col> ' + ' '.join(column_name) + ' </col>', qdmr, data_attributes




# with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/data/break/logical-forms-fixed/train.jsonl") as f:
#     for d in f:
#         qid.append(d["subset_idx"])
#         question.append(input_combine(d["text"], d['column_data']))
#         data.append(qdmr_clean(d["qdmr_code"][0]))
#         # print(d["qdmr_code"])


# # # print(pd.DataFrame(data))
# train_data = [pd.DataFrame(qid), pd.DataFrame(question), pd.DataFrame(data)]
# train_data_header =  ["id","source_text", "target_text"]
# train_data_df = pd.concat(train_data, axis=1, keys=train_data_header)
# train_data_df.to_csv('train_data_df_spider_v2.csv', index=False)


# with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/data/break/logical-forms-fixed/val.jsonl") as f:
#     for d in f:
#         qid.append(d["subset_idx"])
#         question.append(input_combine(d["text"], d['column_data']))
#         data.append(qdmr_clean(d["qdmr_code"][0]))
#         # print(d["qdmr_code"])

# # print(pd.DataFrame(data))
# val_data = [pd.DataFrame(qid), pd.DataFrame(question), pd.DataFrame(data)]
# val_data_header =  ["id","source_text", "target_text"]
# val_data_df = pd.concat(val_data, axis=1, keys=val_data_header)
# val_data_df.to_csv('val_data_df_spider_v2.csv', index=False)

data = []
question = []
qid = []
data_attr = []


# with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/data/break/logical-forms-fixed/train.jsonl") as f:
#     for d in f:
#         qdmr = qdmr_clean(d["qdmr_code"][0])
#         input_text, abs_qdmr, data_attributes = input_abstract(d["text"], d['column_data'], qdmr)
#         qid.append(d["subset_idx"])
#         question.append(input_text)
#         data.append(abs_qdmr)
#         data_attr.append(str(data_attributes))
#         # print(d["qdmr_code"])

# # print(pd.DataFrame(data))
# val_data = [pd.DataFrame(qid), pd.DataFrame(question), pd.DataFrame(data), pd.DataFrame(data_attr)]
# val_data_header =  ["id","source_text", "target_text", "data_attributes"]
# val_data_df = pd.concat(val_data, axis=1, keys=val_data_header)
# val_data_df.to_csv('train_data_df_spider_abstract.csv', index=False)



with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/data/break/logical-forms-fixed/val.jsonl") as f:
    for d in f:
        qdmr = qdmr_clean(d["qdmr_code"][0])
        input_text, abs_qdmr, data_attributes = input_abstract(d["text"], d['column_data'], qdmr)
        qid.append(d["subset_idx"])
        question.append(input_text)
        data.append(abs_qdmr)
        data_attr.append(str(data_attributes))
        # print(d["qdmr_code"])

# print(pd.DataFrame(data))
val_data = [pd.DataFrame(qid), pd.DataFrame(question), pd.DataFrame(data), pd.DataFrame(data_attr)]
val_data_header =  ["id","source_text", "target_text", "data_attributes"]
val_data_df = pd.concat(val_data, axis=1, keys=val_data_header)
val_data_df.to_csv('val_data_df_spider_abstract.csv', index=False)
