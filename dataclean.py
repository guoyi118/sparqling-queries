import jsonlines

data = []
question = []

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
                operation = op[1][2]['arg']['keys'][0] + op[1][2]['arg']['keys'][1]
                attribute = op[1][2]['arg']['keys'][2]['type'] + ':' + op[1][2]['arg']['keys'][2]['keys'][0] + '/' + op[1][2]['arg']['keys'][2]['keys'][1]
                cleaned_code.append(op[0] + ' ' + ref1 + ' '+ ref2 + ' '+ operation + ' ' + attribute )
            else:
                operation = op[1][2]['arg']['keys'][0] + op[1][2]['arg']['keys'][1]
                cleaned_code.append(op[0] + ' ' + ref1 + ' '+ ref2 + ' '+ operation)
        elif op[0] == 'aggregate':
            operator = op[1][0]['arg']
            ref = op[1][1]['arg'][0]
            cleaned_code.append(op[0] + ' ' + operator + ' ' + ref )
        elif op[0] == 'project':
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

        else:
            print(op)
            print(sadsd)
            # cleaned_code.append(op)
    
    return cleaned_code

            



        
        






with jsonlines.open("/home/sdq/GitHub/guoyi/sparqling-queries/text2qdmr/preproc_data/grappa_qdmr_train,emb=bert,cvlink/raw/train.jsonl") as f:
    for d in f:
        question.append(d["text"])
        data.append(d["qdmr_code"][0])
        # print(d["qdmr_code"])


# for i in range(len(data)):
#     try:
#         clean(data[i])
#     except:
#         print(i)
#         print(sddssdds)


# print(question[4])
# print(data[4])
print(clean(data[13]))