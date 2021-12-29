 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **phone** | Name |   |   | 
 | 2 |   | Phone_ID | + |   | 
 | 3 |   | Memory_in_G |   |   | 
 | 4 |   | Carrier |   |   | 
 | 5 |   | Price |   |   | 
 | 6 | **market** | Market_ID | + |   | 
 | 7 |   | District |   |   | 
 | 8 |   | Num_of_employees |   |   | 
 | 9 |   | Num_of_shops |   |   | 
 | 10 |   | Ranking |   |   | 
 | 11 | **phone_market** | Market_ID | + | --> 6 | 
 | 12 |   | Phone_ID |   | --> 2 | 
 | 13 |   | Num_of_stock |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_1978 | How many phones are there? | SELECT count(*) FROM phone | 1. SELECT[tbl:​phone] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​phone] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_1979 | List the names of phones in ascending order of price. | SELECT Name FROM phone ORDER BY Price ASC | 1. SELECT[tbl:​phone] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[col:​phone:​Price, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | 1. SELECT[tbl:​phone] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[col:​phone:​Price, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_1980 | What are the memories and carriers of phones? | SELECT Memory_in_G ,  Carrier FROM phone | 1. SELECT[tbl:​phone] <br>2. PROJECT[col:​phone:​Memory_in_G, #1] <br>3. PROJECT[col:​phone:​Carrier, #1] <br>4. UNION[#2, #3] <br> | 1. SELECT[tbl:​phone] <br>2. PROJECT[col:​phone:​Memory_in_G, #1] <br>3. PROJECT[col:​phone:​Carrier, #1] <br>4. UNION[#2, #3] <br> | + | medium | 
  | SPIDER_train_1981 | List the distinct carriers of phones with memories bigger than 32. | SELECT DISTINCT Carrier FROM phone WHERE Memory_in_G  >  32 | 1. SELECT[col:​phone:​Carrier] <br>2. PROJECT[tbl:​phone, #1] <br>3. PROJECT[col:​phone:​Memory_in_G, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>:​32:​col:​phone:​Memory_in_G] <br>5.*(distinct)* PROJECT[distinct #REF, #4] <br> | 1. SELECT[col:​phone:​Carrier] <br>2. PROJECT[tbl:​phone, #1] <br>3. PROJECT[col:​phone:​Memory_in_G, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>:​32:​col:​phone:​Memory_in_G] <br>5.*(distinct)* PROJECT[None, #4] <br> | + | easy | 
  | SPIDER_train_1982 | Show the names of phones with carrier either "Sprint" or "TMobile". | SELECT Name FROM phone WHERE Carrier  =  "Sprint" OR Carrier  =  "TMobile" | 1. SELECT[tbl:​phone] <br>2. PROJECT[col:​phone:​Carrier, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sprint:​col:​phone:​Carrier] <br>4. COMPARATIVE[#1, #2, comparative:​=:​TMobile:​col:​phone:​Carrier] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​phone:​Name, #5] <br> | 1. SELECT[tbl:​phone] <br>2. PROJECT[col:​phone:​Carrier, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sprint:​col:​phone:​Carrier] <br>4. COMPARATIVE[#1, #2, comparative:​=:​TMobile:​col:​phone:​Carrier] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​phone:​Name, #5] <br> | + | medium | 
  | SPIDER_train_1984 | Show different carriers of phones together with the number of phones with each carrier. | SELECT Carrier ,  COUNT(*) FROM phone GROUP BY Carrier | 1. SELECT[col:​phone:​Carrier] <br>2. PROJECT[tbl:​phone, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | 1. SELECT[col:​phone:​Carrier] <br>2. PROJECT[tbl:​phone, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_train_1987 | Show the names of phones and the districts of markets they are on. | SELECT T3.Name ,  T2.District FROM phone_market AS T1 JOIN market AS T2 ON T1.Market_ID  =  T2.Market_ID JOIN phone AS T3 ON T1.Phone_ID  =  T3.Phone_ID | 1. SELECT[tbl:​phone_market] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[tbl:​phone_market, #1] <br>4. PROJECT[col:​market:​District, #3] <br>5. UNION[#2, #4] <br> | 1. SELECT[tbl:​phone_market] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[tbl:​phone_market, #1] <br>4. PROJECT[col:​market:​District, #3] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_1988 | Show the names of phones and the districts of markets they are on, in ascending order of the ranking of the market. | SELECT T3.Name ,  T2.District FROM phone_market AS T1 JOIN market AS T2 ON T1.Market_ID  =  T2.Market_ID JOIN phone AS T3 ON T1.Phone_ID  =  T3.Phone_ID ORDER BY T2.Ranking | 1. SELECT[tbl:​phone_market] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[tbl:​phone_market, #1] <br>4. PROJECT[col:​market:​District, #3] <br>5. PROJECT[col:​market:​Ranking, #3] <br>6. UNION[#2, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | 1. SELECT[tbl:​phone_market] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[tbl:​phone_market, #1] <br>4. PROJECT[col:​market:​District, #3] <br>5. PROJECT[col:​market:​Ranking, #3] <br>6. UNION[#2, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | + | hard | 
  | SPIDER_train_1990 | For each phone, show its names and total number of stocks. | SELECT T2.Name ,  sum(T1.Num_of_stock) FROM phone_market AS T1 JOIN phone AS T2 ON T1.Phone_ID  =  T2.Phone_ID GROUP BY T2.Name | 1. SELECT[col:​phone:​Name] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[col:​phone_market:​Num_of_stock, #1] <br>4. GROUP[sum, #3, #1] <br>5. UNION[#2, #4] <br> | 1. SELECT[col:​phone:​Name] <br>2. PROJECT[col:​phone:​Name, #1] <br>3. PROJECT[col:​phone_market:​Num_of_stock, #1] <br>4. GROUP[sum, #3, #1] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_1992 | List the names of phones that are not on any market. | SELECT Name FROM phone WHERE Phone_id NOT IN (SELECT Phone_ID FROM phone_market) | 1. SELECT[tbl:​phone] <br>2. FILTER[#1, tbl:​phone_market] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​phone:​Name, #3] <br> | 1. SELECT[tbl:​phone] <br>2. COMPARATIVE[#1, #1, tbl:​phone_market] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​phone:​Name, #3] <br> | + | hard | 
 ***
 Exec acc: **1.0000**
