 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **aircraft** | Aircraft_ID | + |   | 
 | 2 |   | Order_Year |   |   | 
 | 3 |   | Manufacturer |   |   | 
 | 4 |   | Model |   |   | 
 | 5 |   | Fleet_Series |   |   | 
 | 6 |   | Powertrain |   |   | 
 | 7 |   | Fuel_Propulsion |   |   | 
 | 8 | **pilot** | Pilot_ID | + |   | 
 | 9 |   | Pilot_name |   |   | 
 | 10 |   | Rank |   |   | 
 | 11 |   | Age |   |   | 
 | 12 |   | Nationality |   |   | 
 | 13 |   | Position |   |   | 
 | 14 |   | Join_Year |   |   | 
 | 15 |   | Team |   |   | 
 | 16 | **pilot_record** | Record_ID |   |   | 
 | 17 |   | Pilot_ID | + | --> 8 | 
 | 18 |   | Aircraft_ID |   | --> 1 | 
 | 19 |   | Date |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_2083 | How many pilots are there? | SELECT count(*) FROM pilot | 1. SELECT[tbl:​pilot] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​pilot] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_2084 | List the names of pilots in ascending order of rank. | SELECT Pilot_name FROM pilot ORDER BY Rank ASC | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Pilot_name, #1] <br>3. PROJECT[col:​pilot:​Rank, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Pilot_name, #1] <br>3. PROJECT[col:​pilot:​Rank, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_2085 | What are the positions and teams of pilots? | SELECT POSITION ,  Team FROM pilot | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Position, #1] <br>3. PROJECT[col:​pilot:​Team, #1] <br>4. UNION[#2, #3] <br> | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Position, #1] <br>3. PROJECT[col:​pilot:​Team, #1] <br>4. UNION[#2, #3] <br> | + | medium | 
  | SPIDER_train_2086 | List the distinct positions of pilots older than 30. | SELECT DISTINCT POSITION FROM pilot WHERE Age  >  30 | 1. SELECT[tbl:​pilot] <br>2. FILTER[#1, comparative:​>:​30:​col:​pilot:​Age] <br>3. PROJECT[col:​pilot:​Position, #2] <br>4.*(distinct)* PROJECT[distinct #REF, #3] <br> | 1. SELECT[tbl:​pilot] <br>2. COMPARATIVE[#1, #1, comparative:​>:​30:​col:​pilot:​Age] <br>3. PROJECT[col:​pilot:​Position, #2] <br>4.*(distinct)* PROJECT[None, #3] <br> | + | easy | 
  | SPIDER_train_2087 | Show the names of pilots from team "Bradley" or "Fordham". | SELECT Pilot_name FROM pilot WHERE Team  =  "Bradley" OR Team  =  "Fordham" | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Team, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Bradley:​col:​pilot:​Team] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Fordham:​col:​pilot:​Team] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​pilot:​Pilot_name, #5] <br> | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Team, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Bradley:​col:​pilot:​Team] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Fordham:​col:​pilot:​Team] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​pilot:​Pilot_name, #5] <br> | + | medium | 
  | SPIDER_train_2090 | Show the most common nationality of pilots. | SELECT Nationality FROM pilot GROUP BY Nationality ORDER BY COUNT(*) DESC LIMIT 1 | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Nationality, #1] <br>3. GROUP[count, #1, #2] <br>4. SUPERLATIVE[comparative:​max:​None, #2, #3] <br> | 1. SELECT[tbl:​pilot] <br>2. PROJECT[col:​pilot:​Nationality, #1] <br>3. GROUP[count, #1, #2] <br>4. SUPERLATIVE[comparative:​max:​None, #2, #3] <br> | + | hard | 
  | SPIDER_train_2091 | Show the pilot positions that have both pilots joining after year 2005 and pilots joining before 2000. | SELECT POSITION FROM pilot WHERE Join_Year	 <  2000 INTERSECT SELECT POSITION FROM pilot WHERE Join_Year	 >  2005 | 1. SELECT[tbl:​pilot] <br>2. FILTER[#1, comparative:​<:​2000:​col:​pilot:​Join_Year] <br>3. FILTER[#1, comparative:​>:​2005:​col:​pilot:​Join_Year] <br>4. PROJECT[col:​pilot:​Position, #1] <br>5. INTERSECTION[#4, #2, #3] <br> | 1. SELECT[tbl:​pilot] <br>2. COMPARATIVE[#1, #1, comparative:​<:​2000:​col:​pilot:​Join_Year] <br>3. COMPARATIVE[#1, #1, comparative:​>:​2005:​col:​pilot:​Join_Year] <br>4. PROJECT[col:​pilot:​Position, #1] <br>5. INTERSECTION[#4, #2, #3] <br> | + | hard | 
  | SPIDER_train_2092 | Show the names of pilots and models of aircrafts they have flied with. | SELECT T3.Pilot_name ,  T2.Model FROM pilot_record AS T1 JOIN aircraft AS T2 ON T1.Aircraft_ID  =  T2.Aircraft_ID JOIN pilot AS T3 ON T1.Pilot_ID  =  T3.Pilot_ID | 1. SELECT[tbl:​pilot_record] <br>2. PROJECT[tbl:​aircraft, #1] <br>3. PROJECT[col:​pilot:​Pilot_name, #2] <br>4. PROJECT[col:​aircraft:​Model, #1] <br>5. UNION[#3, #4] <br> | 1. SELECT[tbl:​pilot_record] <br>2. PROJECT[tbl:​aircraft, #1] <br>3. PROJECT[col:​pilot:​Pilot_name, #2] <br>4. PROJECT[col:​aircraft:​Model, #1] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_train_2093 | Show the names of pilots and fleet series of the aircrafts they have flied with in ascending order of the rank of the pilot. | SELECT T3.Pilot_name ,  T2.Fleet_Series FROM pilot_record AS T1 JOIN aircraft AS T2 ON T1.Aircraft_ID  =  T2.Aircraft_ID JOIN pilot AS T3 ON T1.Pilot_ID  =  T3.Pilot_ID ORDER BY T3.Rank | 1. SELECT[tbl:​pilot_record] <br>2. PROJECT[col:​pilot:​Pilot_name, #1] <br>3. PROJECT[tbl:​aircraft, #1] <br>4. PROJECT[col:​aircraft:​Fleet_Series, #3] <br>5. PROJECT[col:​pilot:​Rank, #1] <br>6. UNION[#2, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | 1. SELECT[tbl:​pilot_record] <br>2. PROJECT[col:​pilot:​Pilot_name, #1] <br>3. PROJECT[tbl:​aircraft, #1] <br>4. PROJECT[col:​aircraft:​Fleet_Series, #3] <br>5. PROJECT[col:​pilot:​Rank, #1] <br>6. UNION[#2, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | + | hard | 
  | SPIDER_train_2094 | Show the fleet series of the aircrafts flied by pilots younger than 34 | SELECT T2.Fleet_Series FROM pilot_record AS T1 JOIN aircraft AS T2 ON T1.Aircraft_ID  =  T2.Aircraft_ID JOIN pilot AS T3 ON T1.Pilot_ID  =  T3.Pilot_ID WHERE T3.Age  <  34 | 1. SELECT[tbl:​pilot] <br>2. FILTER[#1, comparative:​<:​34:​col:​pilot:​Age] <br>3. PROJECT[tbl:​aircraft, #2] <br>4. PROJECT[col:​aircraft:​Fleet_Series, #3] <br> | 1. SELECT[tbl:​pilot] <br>2. COMPARATIVE[#1, #1, comparative:​<:​34:​col:​pilot:​Age] <br>3. PROJECT[tbl:​aircraft, #2] <br>4. PROJECT[col:​aircraft:​Fleet_Series, #3] <br> | + | hard | 
  | SPIDER_train_2095 | Show the names of pilots and the number of records they have. | SELECT T2.Pilot_name ,  COUNT(*) FROM pilot_record AS T1 JOIN pilot AS T2 ON T1.pilot_ID  =  T2.pilot_ID GROUP BY T2.Pilot_name | 1. SELECT[col:​pilot:​Pilot_name] <br>2. PROJECT[col:​pilot:​Pilot_name, #1] <br>3. PROJECT[tbl:​pilot_record, #1] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | 1. SELECT[col:​pilot:​Pilot_name] <br>2. PROJECT[col:​pilot:​Pilot_name, #1] <br>3. PROJECT[tbl:​pilot_record, #1] <br>4. GROUP[count, #3, #1] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_2097 | List the names of pilots that do not have any record. | SELECT Pilot_name FROM pilot WHERE Pilot_ID NOT IN (SELECT Pilot_ID FROM pilot_record) | 1. SELECT[tbl:​pilot] <br>2. FILTER[#1, tbl:​pilot_record] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​pilot:​Pilot_name, #3] <br> | 1. SELECT[tbl:​pilot] <br>2. COMPARATIVE[#1, #1, tbl:​pilot_record] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​pilot:​Pilot_name, #3] <br> | + | hard | 
 ***
 Exec acc: **1.0000**
