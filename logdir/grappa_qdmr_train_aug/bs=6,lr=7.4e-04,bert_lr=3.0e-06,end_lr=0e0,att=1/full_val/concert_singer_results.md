 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **stadium** | Stadium_ID | + |   | 
 | 2 |   | Location |   |   | 
 | 3 |   | Name |   |   | 
 | 4 |   | Capacity |   |   | 
 | 5 |   | Highest |   |   | 
 | 6 |   | Lowest |   |   | 
 | 7 |   | Average |   |   | 
 | 8 | **singer** | Singer_ID | + |   | 
 | 9 |   | Name |   |   | 
 | 10 |   | Country |   |   | 
 | 11 |   | Song_Name |   |   | 
 | 12 |   | Song_release_year |   |   | 
 | 13 |   | Age |   |   | 
 | 14 |   | Is_male |   |   | 
 | 15 | **concert** | concert_ID | + |   | 
 | 16 |   | concert_Name |   |   | 
 | 17 |   | Theme |   |   | 
 | 18 |   | Stadium_ID |   | --> 1 | 
 | 19 |   | Year |   |   | 
 | 20 | **singer_in_concert** | concert_ID | + | --> 15 | 
 | 21 |   | Singer_ID |   | --> 8 | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_dev_0 | How many singers do we have? | SELECT count(*) FROM singer | 1. SELECT[tbl:​singer] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​singer] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_dev_1 | What is the total number of singers? | SELECT count(*) FROM singer | 1. SELECT[tbl:​singer] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​singer] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_dev_2 | Show name, country, age for all singers ordered by age from the oldest to the youngest. | SELECT name ,  country ,  age FROM singer ORDER BY age DESC | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Name, #1] <br>3. PROJECT[col:​singer:​Country, #1] <br>4. PROJECT[col:​singer:​Age, #1] <br>5. UNION[#2, #3, #4] <br>6. SORT[#5, #4, sortdir:​descending] <br> | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Name, #1] <br>3. PROJECT[col:​singer:​Country, #1] <br>4. PROJECT[col:​singer:​Age, #1] <br>5. UNION[#2, #3, #4] <br>6. SORT[#5, #4, sortdir:​descending] <br> | + | medium | 
  | SPIDER_dev_3 | What are the names, countries, and ages for every singer in descending order of age? | SELECT name ,  country ,  age FROM singer ORDER BY age DESC | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Name, #1] <br>3. PROJECT[col:​singer:​Country, #1] <br>4. PROJECT[col:​singer:​Age, #1] <br>5. UNION[#2, #3, #4] <br>6. SORT[#5, #4, sortdir:​descending] <br> | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Name, #1] <br>3. PROJECT[col:​singer:​Country, #1] <br>4. PROJECT[col:​singer:​Age, #1] <br>5. UNION[#2, #3, #4] <br>6. SORT[#5, #4, sortdir:​descending] <br> | + | medium | 
  | SPIDER_dev_4 | What is the average, minimum, and maximum age of all singers from France? | SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France' | 1. SELECT[tbl:​singer] <br>2. FILTER[#1, comparative:​=:​France:​col:​singer:​Country] <br>3. PROJECT[col:​singer:​Age, #2] <br>4. AGGREGATE[avg, #3] <br>5. AGGREGATE[min, #3] <br>6. AGGREGATE[max, #3] <br>7. UNION[#4, #5, #6] <br> | 1. SELECT[tbl:​singer] <br>2. COMPARATIVE[#1, #1, comparative:​=:​France:​col:​singer:​Country] <br>3. PROJECT[col:​singer:​Age, #2] <br>4. AGGREGATE[avg, #3] <br>5. AGGREGATE[min, #3] <br>6. AGGREGATE[max, #3] <br>7. UNION[#4, #5, #6] <br> | + | medium | 
  | SPIDER_dev_5 | What is the average, minimum, and maximum age for all French singers? | SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France' | 1. SELECT[tbl:​singer] <br>2. FILTER[#1, val:​singer:​Country:​France] <br>3. PROJECT[col:​singer:​Age, #2] <br>4. AGGREGATE[avg, #3] <br>5. AGGREGATE[min, #3] <br>6. AGGREGATE[max, #3] <br>7. UNION[#4, #5, #6] <br> | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Age, #1] <br>3. AGGREGATE[avg, #2] <br>4. AGGREGATE[min, #2] <br>5. AGGREGATE[max, #2] <br>6. UNION[#3, #4, #5] <br> | - | medium | 
  | SPIDER_dev_6 | Show the name and the release year of the song by the youngest singer. | SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1 | 1. SELECT[col:​singer:​Song_Name] <br>2. PROJECT[tbl:​singer, #1] <br>3. PROJECT[col:​singer:​Age, #2] <br>4. SUPERLATIVE[comparative:​min:​None, #1, #3] <br>5. PROJECT[col:​singer:​Song_Name, #4] <br>6. PROJECT[col:​singer:​Song_release_year, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[tbl:​singer] <br>2. PROJECT[tbl:​singer, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​min:​None, #1, #3] <br>5. PROJECT[col:​singer:​Song_Name, #4] <br>6. PROJECT[col:​singer:​Song_release_year, #4] <br>7. UNION[#5, #6] <br> | + | medium | 
  | SPIDER_dev_7 | What are the names and release years for all the songs of the youngest singer? | SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1 | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Age, #1] <br>3. SUPERLATIVE[comparative:​min:​None, #1, #2] <br>4. PROJECT[col:​singer:​Song_Name, #3] <br>5. PROJECT[col:​singer:​Song_Name, #4] <br>6. PROJECT[col:​singer:​Song_release_year, #4] <br>7. UNION[#5, #6] <br> | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Is_male, #1] <br>3. SUPERLATIVE[comparative:​min:​None, #1, #2] <br>4. PROJECT[col:​singer:​Name, #3] <br>5. PROJECT[col:​singer:​Song_release_year, #3] <br>6. UNION[#4, #5] <br> | - | medium | 
  | SPIDER_dev_8 | What are all distinct countries where singers above age 20 are from? | SELECT DISTINCT country FROM singer WHERE age  >  20 | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Age, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​20:​col:​singer:​Age] <br>4.*(distinct)* PROJECT[col:​singer:​Country, #3] <br> | 1. SELECT[tbl:​singer] <br>2. PROJECT[col:​singer:​Age, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​20:​col:​singer:​Age] <br>4. PROJECT[col:​singer:​Country, #3] <br>5.*(distinct)* PROJECT[None, #4] <br> | + | easy | 
  | SPIDER_dev_9 | What are  the different countries with singers above age 20? | SELECT DISTINCT country FROM singer WHERE age  >  20 | 1. SELECT[col:​singer:​Country] <br>2. PROJECT[tbl:​singer, #1] <br>3. PROJECT[col:​singer:​Age, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>:​20:​col:​singer:​Age] <br>5.*(distinct)* PROJECT[different #REF, #4] <br> | 1. SELECT[col:​singer:​Country] <br>2. PROJECT[tbl:​singer, #1] <br>3. PROJECT[col:​singer:​Age, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>:​20:​col:​singer:​Age] <br>5.*(distinct)* PROJECT[None, #4] <br> | + | easy | 
 ***
 Exec acc: **0.8000**
