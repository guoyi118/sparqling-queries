import pandas as pd 
import copy
df = pd.read_csv('/home/sdq/GitHub/guoyi/sparqling-queries/winter_olympic.csv')

input_template = "How many countries medals did <country> win in <sport>? <tbl> winter_olympic </tbl> <col> Country Gold_Medal Silver_Medal Bronze_Medal Total_Medal Sport Continent </col>" 
qdmr_template = "select tbl:winter_olympic, comparative #1 #1 =<country> col:winter_olympic/Country, comparative #1 #1 =<sport> col:winter_olympic/Sport, union #1 #2 #3, project col:winter_olympic/<gold1> #4"

source_text = []
target_text = []
ids = []
for contry in ['Switzerland', 'USA', 'Norway', 'China', 'Sweden']:
    input_text = copy.deepcopy(input_template)
    qdmr_text = copy.deepcopy(qdmr_template)

    for sports in ['Alpine Skiing', 'Cross-Country Skiing', 'Freestyle Skiing', 'Snowboard']:
        
        for medal in ['gold' , 'silver', 'bronze']:
            # print(input_text.replace('<gold>', '123'))
            source_text.append(input_text.replace('<country>',contry).replace('<sport>',sports).replace('<gold>', medal))
            # print(input_text)
            ids.append('1')
            if medal == 'gold':
                target_text.append(qdmr_text.replace('<country>',contry).replace('<sport>',sports).replace('<gold1>','Gold_Medal'))
            if medal == 'silver':
                target_text.append(qdmr_text.replace('<country>',contry).replace('<sport>',sports).replace('<gold1>','Silver_Medal'))
            if medal == 'bronze':
                target_text.append(qdmr_text.replace('<country>',contry).replace('<sport>',sports).replace('<gold1>','Bronze_Medal'))
            

            
        

val_data = [pd.DataFrame(ids), pd.DataFrame(source_text), pd.DataFrame(target_text)]
val_data_header =  ['id',"source_text", "target_text"]
val_data_df = pd.concat(val_data, axis=1, keys=val_data_header)
val_data_df.to_csv('made_data.csv', index=False)




