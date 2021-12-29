 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **roller_coaster** | Roller_Coaster_ID | + |   | 
 | 2 |   | Name |   |   | 
 | 3 |   | Park |   |   | 
 | 4 |   | Country_ID |   | --> 10 | 
 | 5 |   | Length |   |   | 
 | 6 |   | Height |   |   | 
 | 7 |   | Speed |   |   | 
 | 8 |   | Opened |   |   | 
 | 9 |   | Status |   |   | 
 | 10 | **country** | Country_ID | + |   | 
 | 11 |   | Name |   |   | 
 | 12 |   | Population |   |   | 
 | 13 |   | Area |   |   | 
 | 14 |   | Languages |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_6203 | How many roller coasters are there? | SELECT count(*) FROM roller_coaster | 1. SELECT[tbl:​roller_coaster] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_6204 | List the names of roller coasters by ascending order of length. | SELECT Name FROM roller_coaster ORDER BY LENGTH ASC | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Name, #1] <br>3. PROJECT[col:​roller_coaster:​Length, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Name, #1] <br>3. PROJECT[col:​roller_coaster:​Length, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_6205 | What are the lengths and heights of roller coasters? | SELECT LENGTH ,  Height FROM roller_coaster | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Length, #1] <br>3. PROJECT[col:​roller_coaster:​Height, #1] <br>4. UNION[#2, #3] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Length, #1] <br>3. PROJECT[col:​roller_coaster:​Height, #1] <br>4. UNION[#2, #3] <br> | + | medium | 
  | SPIDER_train_6206 | List the names of countries whose language is not "German". | SELECT Name FROM country WHERE Languages != "German" | 1. SELECT[tbl:​country] <br>2. PROJECT[col:​country:​Languages, #1] <br>3. COMPARATIVE[#1, #2, comparative:​!=:​German:​col:​country:​Languages] <br>4. PROJECT[col:​country:​Name, #3] <br> | 1. SELECT[tbl:​country] <br>2. PROJECT[col:​country:​Languages, #1] <br>3. COMPARATIVE[#1, #2, comparative:​!=:​German:​col:​country:​Languages] <br>4. PROJECT[col:​country:​Name, #3] <br> | + | easy | 
  | SPIDER_train_6209 | What is the average speed of roller coasters? | SELECT avg(Speed) FROM roller_coaster | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Speed, #1] <br>3. AGGREGATE[avg, #2] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Speed, #1] <br>3. AGGREGATE[avg, #2] <br> | + | easy | 
  | SPIDER_train_6210 | Show the different statuses and the numbers of roller coasters for each status. | SELECT Status ,  COUNT(*) FROM roller_coaster GROUP BY Status | 1.*(distinct)* SELECT[col:​roller_coaster:​Status] <br>2. PROJECT[tbl:​roller_coaster, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | 1.*(distinct)* SELECT[col:​roller_coaster:​Status] <br>2. PROJECT[tbl:​roller_coaster, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_train_6211 | Please show the most common status of roller coasters. | SELECT Status FROM roller_coaster GROUP BY Status ORDER BY COUNT(*) DESC LIMIT 1 | 1. SELECT[col:​roller_coaster:​Status] <br>2. PROJECT[tbl:​roller_coaster, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​roller_coaster:​Status] <br>2. PROJECT[tbl:​roller_coaster, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_6212 | List the status shared by more than two roller coaster. | SELECT Status FROM roller_coaster GROUP BY Status HAVING COUNT(*)  >  2 | 1. SELECT[col:​roller_coaster:​Status] <br>2. PROJECT[tbl:​roller_coaster, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>:​2] <br> | 1. SELECT[col:​roller_coaster:​Status] <br>2. PROJECT[tbl:​roller_coaster, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>:​2] <br> | + | easy | 
  | SPIDER_train_6213 | Show the park of the roller coaster with the highest speed. | SELECT Park FROM roller_coaster ORDER BY Speed DESC LIMIT 1 | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Speed, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​roller_coaster:​Park, #3] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Speed, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​roller_coaster:​Park, #3] <br> | + | medium | 
  | SPIDER_train_6214 | Show the names of roller coasters and names of country they are in. | SELECT T2.Name ,  T1.Name FROM country AS T1 JOIN roller_coaster AS T2 ON T1.Country_ID  =  T2.Country_ID | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Name, #1] <br>3. PROJECT[tbl:​country, #1] <br>4. PROJECT[col:​country:​Name, #3] <br>5. UNION[#2, #4] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[col:​roller_coaster:​Name, #1] <br>3. PROJECT[tbl:​country, #1] <br>4. PROJECT[col:​country:​Name, #3] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_6215 | Show the names of countries that have more than one roller coaster. | SELECT T1.Name FROM country AS T1 JOIN roller_coaster AS T2 ON T1.Country_ID  =  T2.Country_ID GROUP BY T1.Name HAVING COUNT(*)  >  1 | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[tbl:​country, #1] <br>3. GROUP[count, #1, #2] <br>4. COMPARATIVE[#2, #3, comparative:​>:​1] <br>5. PROJECT[col:​country:​Name, #4] <br> | 1. SELECT[tbl:​roller_coaster] <br>2. PROJECT[tbl:​country, #1] <br>3. GROUP[count, #1, #2] <br>4. COMPARATIVE[#2, #3, comparative:​>:​1] <br>5. PROJECT[col:​country:​Name, #4] <br> | + | medium | 
  | SPIDER_train_6217 | Show the names of countries and the average speed of roller coasters from each country. | SELECT T1.Name ,  avg(T2.Speed) FROM country AS T1 JOIN roller_coaster AS T2 ON T1.Country_ID  =  T2.Country_ID GROUP BY T1.Name | 1. SELECT[col:​country:​Name] <br>2. PROJECT[col:​country:​Name, #1] <br>3. PROJECT[tbl:​roller_coaster, #1] <br>4. PROJECT[col:​roller_coaster:​Speed, #3] <br>5. GROUP[avg, #4, #1] <br>6. UNION[#2, #5] <br> | 1. SELECT[col:​country:​Name] <br>2. PROJECT[col:​country:​Name, #1] <br>3. PROJECT[tbl:​roller_coaster, #1] <br>4. PROJECT[col:​roller_coaster:​Speed, #3] <br>5. GROUP[avg, #4, #1] <br>6. UNION[#2, #5] <br> | + | medium | 
 ***
 Exec acc: **1.0000**
