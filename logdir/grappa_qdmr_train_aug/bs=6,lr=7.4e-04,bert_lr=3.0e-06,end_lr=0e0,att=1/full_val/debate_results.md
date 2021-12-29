 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **people** | People_ID | + |   | 
 | 2 |   | District |   |   | 
 | 3 |   | Name |   |   | 
 | 4 |   | Party |   |   | 
 | 5 |   | Age |   |   | 
 | 6 | **debate** | Debate_ID | + |   | 
 | 7 |   | Date |   |   | 
 | 8 |   | Venue |   |   | 
 | 9 |   | Num_of_Audience |   |   | 
 | 10 | **debate_people** | Debate_ID | + | --> 6 | 
 | 11 |   | Affirmative |   | --> 1 | 
 | 12 |   | Negative |   | --> 1 | 
 | 13 |   | If_Affirmative_Win |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_1492 | How many debates are there? | SELECT count(*) FROM debate | 1. SELECT[tbl:​debate] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​debate] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_1493 | List the venues of debates in ascending order of the number of audience. | SELECT Venue FROM debate ORDER BY Num_of_Audience ASC | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Venue, #1] <br>3. PROJECT[col:​debate:​Num_of_Audience, #1] <br>4. GROUP[count, #3, #1] <br>5. SORT[#2, #4, sortdir:​ascending] <br> | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Venue, #1] <br>3. PROJECT[col:​debate:​Num_of_Audience, #1] <br>4. GROUP[count, #3, #1] <br>5. SORT[#2, #4, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_1494 | What are the date and venue of each debate? | SELECT Date ,  Venue FROM debate | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Date, #1] <br>3. PROJECT[col:​debate:​Venue, #1] <br>4. UNION[#2, #3] <br> | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Date, #1] <br>3. PROJECT[col:​debate:​Venue, #1] <br>4. UNION[#2, #3] <br> | + | medium | 
  | SPIDER_train_1495 | List the dates of debates with number of audience bigger than 150 | SELECT Date FROM debate WHERE Num_of_Audience  >  150 | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Num_of_Audience, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​150:​col:​debate:​Num_of_Audience] <br>4. PROJECT[col:​debate:​Date, #3] <br> | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Num_of_Audience, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​150:​col:​debate:​Num_of_Audience] <br>4. PROJECT[col:​debate:​Date, #3] <br> | + | easy | 
  | SPIDER_train_1496 | Show the names of people aged either 35 or 36. | SELECT Name FROM  people WHERE Age  =  35 OR Age  =  36 | 1. SELECT[tbl:​people] <br>2. FILTER[#1, comparative:​=:​35:​col:​people:​Age] <br>3. FILTER[#1, comparative:​=:​36:​col:​people:​Age] <br>4. UNION[#2, #3] <br>5. PROJECT[col:​people:​Name, #4] <br> | 1. SELECT[tbl:​people] <br>2. COMPARATIVE[#1, #1, comparative:​=:​35:​col:​people:​Age] <br>3. COMPARATIVE[#1, #1, comparative:​=:​36:​col:​people:​Age] <br>4. UNION[#2, #3] <br>5. PROJECT[col:​people:​Name, #4] <br> | + | medium | 
  | SPIDER_train_1498 | Show different parties of people along with the number of people in each party. | SELECT Party ,  COUNT(*) FROM people GROUP BY Party | 1. SELECT[col:​people:​Party] <br>2. PROJECT[tbl:​people, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | 1. SELECT[col:​people:​Party] <br>2. PROJECT[tbl:​people, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_train_1499 | Show the party that has the most people. | SELECT Party FROM people GROUP BY Party ORDER BY COUNT(*) DESC LIMIT 1 | 1. SELECT[col:​people:​Party] <br>2. PROJECT[tbl:​people, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​people:​Party] <br>2. PROJECT[tbl:​people, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_1500 | Show the distinct venues of debates | SELECT DISTINCT Venue FROM debate | 1. SELECT[col:​debate:​Venue] <br>2. FILTER[#1, tbl:​debate] <br>3.*(distinct)* PROJECT[distinct #REF, #2] <br> | 1. SELECT[col:​debate:​Venue] <br>2. COMPARATIVE[#1, #1, tbl:​debate] <br>3.*(distinct)* PROJECT[None, #2] <br> | + | easy | 
  | SPIDER_train_1503 | Show the names of people that are on affirmative side of debates with number of audience bigger than 200. | SELECT T3.Name FROM debate_people AS T1 JOIN debate AS T2 ON T1.Debate_ID  =  T2.Debate_ID JOIN people AS T3 ON T1.Affirmative  =  T3.People_ID WHERE T2.Num_of_Audience  >  200 | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Num_of_Audience, #1] <br>3. PROJECT[size of #REF, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>:​200:​col:​debate:​Num_of_Audience] <br>5. PROJECT[col:​debate_people:​Affirmative, #4] <br>6. PROJECT[tbl:​debate_people, #5] <br>7. PROJECT[col:​people:​Name, #6] <br> | 1. SELECT[tbl:​debate] <br>2. PROJECT[col:​debate:​Num_of_Audience, #1] <br>3. PROJECT[None, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>:​200:​col:​debate:​Num_of_Audience] <br>5. PROJECT[col:​debate_people:​Affirmative, #4] <br>6. PROJECT[tbl:​debate_people, #5] <br>7. PROJECT[col:​people:​Name, #6] <br> | + | hard | 
  | SPIDER_train_1504 | Show the names of people and the number of times they have been on the affirmative side of debates. | SELECT T2.Name ,  COUNT(*) FROM debate_people AS T1 JOIN people AS T2 ON T1.Affirmative  =  T2.People_ID GROUP BY T2.Name | 1. SELECT[tbl:​debate_people] <br>2. PROJECT[tbl:​debate_people, #1] <br>3. FILTER[#2, col:​debate_people:​Affirmative] <br>4. PROJECT[col:​people:​Name, #1] <br>5. GROUP[count, #3, #4] <br>6. UNION[#4, #5] <br> | 1. SELECT[tbl:​debate_people] <br>2. PROJECT[tbl:​debate_people, #1] <br>3. COMPARATIVE[#2, #2, col:​debate_people:​Affirmative] <br>4. PROJECT[col:​people:​Name, #1] <br>5. GROUP[count, #3, #4] <br>6. UNION[#4, #5] <br> | + | medium | 
 ***
 Exec acc: **1.0000**
